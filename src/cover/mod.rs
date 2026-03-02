use crate::bitsets::{BitCollection, Bitset, BitsetInit};
use crate::cover::reversible_cover::{ShallowBitset, SparseBitset};
use crate::globals::{attribute, item_type};

pub mod reversible_cover;
pub mod similarities;

pub struct Cover {
    pub num_attributes: usize,
    pub num_labels: usize,
    attributes: Vec<Bitset>,
    labels: Vec<Bitset>,
    cover: SparseBitset,
    branch: Vec<usize>,
}

impl Cover {
    pub fn new(attributes: Vec<Bitset>, labels: Vec<Bitset>, size: usize) -> Self {
        Self {
            num_attributes: attributes.len(),
            num_labels: labels.len(),
            attributes,
            labels,
            cover: SparseBitset::new(size),
            branch: vec![],
        }
    }

    pub fn count(&self) -> usize {
        self.cover.count()
    }

    pub fn labels_count(&self) -> Vec<usize> {
        self.cover.count_intersect_with_many(&self.labels)
    }

    // TODO : Implementation to use when using a placeholder for hot activities
    pub fn labels_count_with_buffer(&self, buffer: &mut Vec<usize>) {
        buffer.clear();
        buffer.extend_from_slice(&self.cover.count_intersect_with_many(&self.labels));
    }

    pub fn branch_on(&mut self, item: usize) -> usize {
        self.branch.push(item);
        let attribute = attribute(item);
        let invert = item_type(item) == 0;
        self.cover
            .intersect_with(&self.attributes[attribute], invert)
    }

    pub fn count_if_branch_on(&self, item: usize) -> usize {
        let attribute = attribute(item);
        let invert = item_type(item) == 0;
        self.cover
            .count_intersect_with(&self.attributes[attribute], invert)
    }

    pub fn backtrack(&mut self) {
        assert_ne!(self.branch.len(), 0, "No backtrack when at root");
        self.branch.pop();
        self.cover.restore();
    }

    #[inline]
    pub fn to_vec(&self) -> Vec<usize> {
        self.cover.to_vec()
    }

    pub fn shallow_cover(&self) -> ShallowBitset {
        let cover = &self.cover;
        cover.into()
    }

    pub fn sparse(&self) -> &SparseBitset {
        &self.cover
    }

    pub fn path(&self) -> &[usize] {
        &self.branch
    }

    /// Build a Cover directly from binary feature and label arrays.
    ///
    /// `features`: 2-D array of shape (n_samples, n_features) with values 0 or 1.
    /// `labels`:   1-D array of length n_samples with non-negative integer class labels.
    ///
    /// Returns `(cover, label_map)` where `label_map[i]` is the original label for
    /// internal class index `i`, in the order the classes were first seen.
    pub fn from_arrays(
        features: ndarray::ArrayView2<i64>,
        labels: ndarray::ArrayView1<i64>,
    ) -> Result<(Self, Vec<i64>), String> {
        let n_samples = features.nrows();
        let n_features = features.ncols();

        if labels.len() != n_samples {
            return Err(format!(
                "X has {} rows but y has {} elements",
                n_samples,
                labels.len()
            ));
        }

        // Build one Bitset per feature column.
        let mut attributes: Vec<Bitset> = Vec::with_capacity(n_features);
        for col in 0..n_features {
            let mut bs = Bitset::new(BitsetInit::Empty(n_samples));
            for row in 0..n_samples {
                let v = features[[row, col]];
                if v == 1 {
                    bs.set(row);
                } else if v != 0 {
                    return Err(format!(
                        "Non-binary value {} at row {}, col {}",
                        v, row, col
                    ));
                }
            }
            attributes.push(bs);
        }

        // Map labels to 0..n_classes in first-seen order.
        let mut label_map: Vec<i64> = Vec::new();
        let mut label_index: std::collections::HashMap<i64, usize> =
            std::collections::HashMap::new();
        let mut normalized: Vec<usize> = Vec::with_capacity(n_samples);
        for &lbl in labels.iter() {
            if lbl < 0 {
                return Err(format!("Negative label {} is not supported", lbl));
            }
            let n = label_map.len();
            let idx = *label_index.entry(lbl).or_insert_with(|| {
                label_map.push(lbl);
                n
            });
            normalized.push(idx);
        }

        let n_classes = label_map.len();
        let mut label_bitsets: Vec<Bitset> =
            vec![Bitset::new(BitsetInit::Empty(n_samples)); n_classes];
        for (sample_idx, &cls) in normalized.iter().enumerate() {
            label_bitsets[cls].set(sample_idx);
        }

        Ok((Cover::new(attributes, label_bitsets, n_samples), label_map))
    }
}
