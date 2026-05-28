use std::collections::HashMap;

/// Character-level tokenizer.
#[derive(Debug, Clone)]
pub struct CharTokenizer {
    stoi: HashMap<char, u32>,
    itos: Vec<char>,
}

impl CharTokenizer {
    pub fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        chars.sort_unstable();
        let stoi = chars
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i as u32))
            .collect();
        Self { stoi, itos: chars }
    }

    pub fn vocab_size(&self) -> usize {
        self.itos.len()
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.chars()
            .filter_map(|c| self.stoi.get(&c).copied())
            .collect()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|&i| self.itos.get(i as usize).copied())
            .collect()
    }
}
