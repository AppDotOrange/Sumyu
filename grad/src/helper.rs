use std::collections::HashMap;
use crate::fnn_lm::tokenize;

pub fn xor() -> Vec<(Vec<f64>, Vec<f64>)> {
    vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ]
}

pub fn char_level_vocab_v1() -> Vec<String> {
    vec!["a", "b", "c", "d", "e",
         "f", "g", "h", "i", "j",
         "k", "l", "m", "n", "o",
         "p", "q", "r", "s", "t",
         "u", "v", "w", "x", "y",
         "z",
         "A", "B", "C", "D", "E",
         "F", "G", "H", "I", "J",
         "K", "L", "M", "N", "O",
         "P", "Q", "R", "S", "T",
         "U", "V", "W", "X", "Y",
         "Z",
         "0", "1", "2", "3", "4",
         "5", "6", "7", "8", "9",
         ".", ",", "?", "!", "+",
         "-", "*", "/", "'", "\"",
         "#", "$", "%", "&", "(",
         ")", "_", ";", ":", "<",
         ">", " ", "\n",].iter().map(|x| { String::from(*x) }).collect()
}

pub fn token_vocab_v1() -> Vec<String> {
    vec!["<UNKNOWN>", "<USER>", "<BOT>", "<EOT>",
         "a", "b", "c", "d", "e",
         "f", "g", "h", "i", "j",
         "k", "l", "m", "n", "o",
         "p", "q", "r", "s", "t",
         "u", "v", "w", "x", "y",
         "z",
         "A", "B", "C", "D", "E",
         "F", "G", "H", "I", "J",
         "K", "L", "M", "N", "O",
         "P", "Q", "R", "S", "T",
         "U", "V", "W", "X", "Y",
         "Z",
         "0", "1", "2", "3", "4",
         "5", "6", "7", "8", "9",
         ".", ",", "?", "!", "+",
         "-", "*", "/", "'", "\"",
         "#", "$", "%", "&", "(",
         ")", "_", ";", ":", "<",
         ">", " ", "\n",
         "th", "the", "he", "she", "it",
         "is", "are", "at", "in", "his",
         "her", "him", "Th", "The", "He",
         "She", "It",].iter().map(|x| { String::from(*x) }).collect()
}

pub fn general_vocab_v1() -> Vec<String> {
     vec!["<UNKNOWN>",
          "a", "b", "c", "d", "e",
          "f", "g", "h", "i", "j",
          "k", "l", "m", "n", "o",
          "p", "q", "r", "s", "t",
          "u", "v", "w", "x", "y",
          "z",
          "A", "B", "C", "D", "E",
          "F", "G", "H", "I", "J",
          "K", "L", "M", "N", "O",
          "P", "Q", "R", "S", "T",
          "U", "V", "W", "X", "Y",
          "Z",
          "0", "1", "2", "3", "4",
          "5", "6", "7", "8", "9",
          ".", ",", "?", "!", "+",
          "-", "*", "/", "'", "\"",
          "#", "$", "%", "&", "(",
          ")", "_", ";", ":", "<",
          ">", " ", "\n", "\\",
          "th", "the", "he", "she", "it",
          "is", "are", "at", "in", "his",
          "her", "him", "Th", "The", "He",
          "She", "It",].iter().map(|x| { String::from(*x) }).collect()
}

pub fn zen_of_python_20_tok_vocab() -> Vec<String> {
     ["<UNKNOWN>", "T", "h", "e", " ",
     "Z", "n", "o", "f", "P",
     "y", "t", ",", "b", "i",
     "m", "r", "s", "\n", "B",
     "a", "u", "l", "g", ".",
     "E", "x", "p", "c", "S",
     "C", "d", "F", "R", "'",
     "k", "A", "v", "U", "I",
     "-", "w", "D", "N", "*",
     "!", "s ", " t", " i", ".\n",
     " th", "er", "e ", "pl", "ou",
     "et", "y ", "on", "n ", "ea",
     " tha", "ter", "ter tha", "ter than ", "etter than ",
     "better than "].iter().map(|x| { String::from(*x) }).collect()
}

pub fn zen_of_python_50_tok_vocab() -> Vec<String> {
     ["<UNKNOWN>", "T", "h", "e", " ",
     "Z", "n", "o", "f", "P",
     "y", "t", ",", "b", "i",
     "m", "r", "s", "\n", "B",
     "a", "u", "l", "g", ".",
     "E", "x", "p", "c", "S",
     "C", "d", "F", "R", "'",
     "k", "A", "v", "U", "I",
     "-", "w", "D", "N", "*",
     "!", "s ", " t", " i", ".\n",
     " th", "er", "e ", "pl", "ou",
     "et", "y ", "on", "n ", "ea",
     " tha", "ter", "ter tha", "ter than ", "etter than ",
     "better than ", " is ", "mpl", "it", "en",
     "th", "t ", "o ", "es", "ti",
     "gh", "ex", "ough", "of", "impl",
     "ic", "ay ", "ar", " is better than ", "y.\n",
     "ver", "tion", "though", "te", "ta",
     "tation", "s better than ", "ompl", "ne", "never",
     "ly "].iter().map(|x| { String::from(*x) }).collect()
}

pub fn zen_of_python_100_tok_vocab() -> Vec<String> {
     ["<UNKNOWN>", "T", "h", "e", " ",
     "Z", "n", "o", "f", "P",
     "y", "t", ",", "b", "i",
     "m", "r", "s", "\n", "B",
     "a", "u", "l", "g", ".",
     "E", "x", "p", "c", "S",
     "C", "d", "F", "R", "'",
     "k", "A", "v", "U", "I",
     "-", "w", "D", "N", "*",
     "!", "s ", " t", " i", ".\n",
     " th", "er", "e ", "pl", "ou",
     "et", "y ", "on", "n ", "ea",
     " tha", "ter", "ter tha", "ter than ", "etter than ",
     "better than ", " is ", "mpl", "it", "en",
     "th", "t ", "o ", "es", "ti",
     "gh", "ex", "ough", "of", "impl",
     "ic", "ay ", "ar", " is better than ", "y.\n",
     "ver", "tion", "though", "te", "ta",
     "tation", "s better than ", "ompl", "ne", "never",
     "ly ", "lthough", "les", "is better than ", "in",
     "il", "icit", "expl", "dea", "d.\n",
     "d ", "to ", "idea", "be ", "al",
     "ac", "Although", ".\nAlthough", "--", " the ",
     "y.\nE", "way ", "vi", "viou", "vious ",
     "to expl", "to expla", "to explain", "to explain,", "to explain, i",
     "the ", "ted.\n", "tation is ", "sil", "silen",
     "sh", "shou", "shoul", "should ", "se ",
     "se", "s.\n", "rea", "re ", "re",
     "ref", "pe", "pec", "peci", "pecial",
     "pecial "].iter().map(|x| { String::from(*x) }).collect()
}

pub fn zen_of_python_100_tok_vocab_v2() -> Vec<String> {
     ["<UNKNOWN>", "T", "h", "e", " ",
     "Z", "n", "o", "f", "P",
     "y", "t", ",", "b", "i",
     "m", "r", "s", "\n", "B",
     "a", "u", "l", "g", ".",
     "E", "x", "p", "c", "S",
     "C", "d", "F", "R", "'",
     "k", "A", "v", "U", "I",
     "-", "w", "D", "N", "*",
     "!", " t", "s ", " i", " th",
     ".\n", " is ", "er", "e ", " tha",
     " than", " than ", "pl", "ter", "tter",
     "etter", "better", "ou", "y ", "on",
     "ea", "mpl", "en", "it", "o ",
     "impl", "es", "at", " the ", "hou",
     "ic", "t ", "gh", "ex", "thou",
     "though", "lthough", "Although", " to ", "Although ",
     "ay ", "expl", "icit", "ompl", "ly ",
     "ar", "dea", "ion", "ation", "tation",
     "ver", "ever", "never", "be ", "y.\n",
     "entation", "mentation", "ementation", "implementation", "d.\n",
     "ed.\n", "of", "les", " idea", " idea.\n",
     "il", "ilen", "d ", "idea", "to ",
     "compl", "shou", "ac", "shoul", "should ",
     "idea.\n", "al", "less ", "may ", "--",
     "th", "ity ", "silen", "in", "ous ",
     "expla", "explain", "explain,", "one ", "way ",
     "ly.\n", "nless ", "ious ", "vious ", "bvious ",
     "obvious ", "se ", "-- ", "at ", "is ",
     "re "].iter().map(|x| { String::from(*x) }).collect()
}

pub fn ml_200_tok_vocab_v1() -> Vec<String> { // trained on this codebase!
     ["<UNKNOWN>", "u", "s", "e", " ", "c", "r", "a", "t", ":", "f", "n", "_", "l", "m", "L", "M", ";", "\r", "\n", "d", "i", "o", "p", "b", "C", "h", "F", "N", "{", "}", "w", "(", ")", "-", ">", "S", "&", ",", "x", "g", "z", "=", "\"", "<", "U", "E", "R", ".", "*", "O", "T", "B", "!", "D", "v", "V", "0", "1", "/", "6", "4", "q", "|", "[", "]", "y", "H", "P", "k", "+", "2", "G", "j", "I", "3", "?", "\\", "Ж", "Х", "A", "J", "K", "Q", "W", "X", "Y", "Z", "5", "7", "8", "9", "'", "#", "$", "%", "^", "  ", "    ", "        ", "\n        ", "\r\n        ", " \"", "\",", "\n    ", "\n                ", "\r\n    ", "\r\n            ", "in", "t ", "en", "()", "er", " {", "se", "or", "at", "le", "let ", "= ", "pu", "\r\n     \"", "sel", "self", "on", "\r\n", " = ", ";\r\n        ", "\n                    ", "\",\r\n        ", "self.", ": ", "ra", " {\r\n            ", "ec", ");\r\n        ", ", ", "::", "re", "ata", "data", "ens", "ensor", "Tensor", " {\r\n        ", "\n            ", "iz", "ize", "size", "Vec", "Vec<", "ch", "ing", "mu", "mut ", "usize", "pub", "pub ", "ter", "}\r\n", "();\r\n        ", "t_", "rad", "grad", "\r\n    pub ", "ou", "th", "n ", "fn ", "pre", "prev", "one", "\n                        ", "inpu", "ro", ",\r\n            ", "\n        let ", "ex", "().", "\r\n            .", " -", "\r\n    }\r\n", "lone", "ar", "ne", "ma", " ->", " -> ", "id", " \", ", "clone", "tr", ".clone", "\r\n\r\n        ", ";\r\n\r\n        ", "(&", "(&self", "to", "\",\r\n          ", "ts", "len", "tring", ".clone()", "iter", ";\n                    ", "inne", "inner", "new", "be", "orro", "orrow", "borrow", ".borrow", "par", "sh", ".iter", ".iter().", "String", "}\r\n        ", "ca", "/ ", "// ", " th", "oca", "um", "im", "for", "tex", "Vec<Tensor", "Vec<Tensor>", "ontex", "contex", "len()", ".len()", ");\r\n            ", "ocab", "vocab", "ding", "dding", "bedding", "mbedding", "embedding", " {\r\n                ", ".borrow_", ".borrow_mu", ".borrow_mut", "vec", "\n                            ", "grad ", "s ", "\n                }", "ion", "for ", "gh", "push", ".push", "para", "param", ".push(", "::new", "\n\n        ", ";\n\n        ", "64", "f64", ".0", ".borrow_mut().", ".borrow_mut().grad ", "prev[", "atch", "batch", "inner.clone()", " {\n                    ", "el", "co", "\r\n                ", "p(", "l\",", "dim", "\n    }", "yer", "ayer", "layer", "(&self, ", ",\r\n        ", "data()", "\n                }\n                ", "de", "].borrow_mut().grad ", "map(", "map(|", ".iter().map(|", "inputs", " in", " in ", "use", "n\",", "ct"].iter().map(|x| { String::from(*x) }).collect()
}

pub fn ml_200_tok_vocab_v2() -> Vec<String> {
     ["<UNKNOWN>", "u", "s", "e", " ", "r", "a", "n", "d", "_", "i", "t", ":", "{", "D", "b", "o", ",", "N", "m", "l", "}", ";", "\r", "\n", "z", "S", "c", "T", "#", "[", "v", "(", "C", ")", "]", "p", "E", "g", "V", "<", "f", "6", "4", ">", "w", "-", "=", "0", ".", "1", "/", "q", "|", "&", "*", "h", "y", "x", "H", "M", "L", "P", "3", "2", "k", "O", "+", "B", "G", "j", "U", "I", "!", "\"", "?", "Ж", "Х", "R", "A", "F", "^", "W", "Y", "7", "5", "\\", "Q", "%", "9", "  ", "    ", "        ", "\n        ", "\r\n        ", "\r\n            ", "\n                ", "in", "\r\n", "er", "t ", "\r\n                ", "()", "en", "\r\n    ", " {", "se", "= ", "or", "le", "let ", "sel", "self", "at", "pu", ": ", "self.", " = ", "ra", "\n                        ", "\n                    ", "on", "\r\n        let ", "::", ", ", "\r\n                    ", "ed", "iz", "ize", "size", "ens", "ensor", "Tensor", "dat", "data", "ing", "ch", "\n            ", "re", "rad", "grad", "ec", ",\r\n            ", " {\r\n            ", ");", "pub", "pub ", "\r\n    pub ", "\r\n\r\n    pub ", "mu", "mut ", "usize", "();", "ro", "\r\n    }", "it", "\n    ", "n ", "Vec", "put", "input", "\r\n        }", "er.", "s: ", "inn", "inner.", "pre", "Vec<", "prev", "fn ", "al", "ding", "edding", "bedding", "mbedding", "embedding", "ne", "\r\n                .", "\n        let ", " -", "one", "\r\n            .", "inner.prev", " {\r\n        ", "().", " ->", " -> ", "iter", "atch", "id", "st", "ma", "s.", "(&", "(&self", "lone", "ar", "\r\n\r\n        ", "len", "_size", "clone", "inner.prev[", "ex", "yer", "ayer", "layer", "par", ";\n                    ", "new", "um", "co", "nt", "batch", "lo", "av", "aved", "orro", "orrow", "borrow", "rain", " {\r\n                ", "for", "_len", ".borrow", "/ ", "// ", "::new", "cont", "contex", "context", "\n                            ", "para", "param", "di", "grad ", "();\r\n\r\n        ", "clone()", "for ", "\n                }", " {\n                    ", "s ", "el", "Vec<Tensor", "Vec<Tensor>", "context_len", ",\r\n    ", ",\r\n        ", "uro", "p(", ");\r\n            ", "ct", "iter()", "map(", "map(|", "och", "poch", "epoch", "dim", "to", "len()", "inner.clone()", "embedding_", "embedding_dim", "in ", " in ", "64", "f64", "\n\n        ", "lect", "llect", "collect", "sh", "\n                }\n                ", "Saved", "(&self)", "+= ", "\r\n            }", ";\r\n\r\n        ", "de", ");\r\n\r\n        ", "uron", "neuron", "::new(", "ig", "igh", "ight", "eight", "weight", "\n    }", ".borrow_"].iter().map(|x| { String::from(*x) }).collect()
}

pub fn ml_200_tok_vocab_v3() -> Vec<String> {
     ["<UNKNOWN>", "u", "s", "e", " ", "r", "a", "n", "d", "_", "i", "t", ":", "{", "D", "b", "o", ",", "N", "m", "l", "}", ";", "\r", "\n", "z", "S", "c", "T", "#", "[", "v", "(", "C", ")", "]", "p", "E", "g", "V", "<", "f", "6", "4", ">", "w", "-", "=", "0", ".", "1", "/", "q", "|", "&", "*", "h", "y", "x", "H", "M", "L", "P", "3", "2", "k", "O", "+", "B", "G", "j", "U", "I", "!", "\"", "?", "Ж", "Х", "R", "A", "F", "^", "W", "Y", "7", "5", "\\", "Q", "%", "9", "  ", "    ", "        ", "\n        ", "\r\n        ", "\r\n            ", "\n                ", "in", "\r\n", "er", "t ", "\r\n                ", "()", "en", "\r\n    ", " {", "se", "= ", "or", "le", "let ", "sel", "self", "at", "pu", ": ", "self.", " = ", "ra", "\n                        ", "\n                    ", "on", "\r\n        let ", "::", ", ", "\r\n                    ", "ed", "iz", "ize", "size", "ens", "ensor", "Tensor", "dat", "data", "ing", "ch", "re", "\n            ", "ec", ",\r\n            ", "rad", "grad", " {\r\n            ", ");", "pub", "pub ", "\r\n    pub ", "\r\n\r\n    pub ", "mu", "mut ", "usize", "();", "\r\n    }", "ro", "t_", "\n    ", "n ", "Vec", "\r\n        }", "er.", "it", "pre", "inn", "inner.", "s: ", "inpu", "Vec<", "fn ", "prev", "al", "ding", "edding", "bedding", "mbedding", "embedding", "ne", "\r\n                .", "\n        let ", " -", "one", "\r\n            .", "inner.prev", "().", " {\r\n        ", " ->", " -> ", "atch", "id", "iter", "ma", "(&", "(&self", "s.", "lone", "ar", "len", "\r\n\r\n        ", "clone", "inner.prev[", "ex", "yer", "ayer", "layer", ";\n                    ", "par", "st", "new", "um", "co", "nt", "lo", "batch", "av", "aved", "orro", "orrow", "borrow", "rain", "input", "_size", " {\r\n                ", ".borrow", "/ ", "// ", "for", "::new", "cont", "contex", "para", "param", "\n                            ", "();\r\n\r\n        ", "grad ", "di", "clone()", "for ", " {\n                    ", "\n                }", "s ", "el", ",\r\n    ", ",\r\n        ", "Vec<Tensor", "Vec<Tensor>", "p(", ");\r\n            ", "context_", "context_len", "uro", "iter()", "ct", "dim", "len()", "to", "och", "poch", "epoch", "map(", "map(|", "inner.clone()", "embedding_", "embedding_dim", "lect", "llect", "collect", "64", "f64", "\n\n        ", "in ", " in ", "\n                }\n                ", "sh", "(&self)", "\r\n            }", "+= ", "Saved", "de", ";\r\n\r\n        ", ".borrow_", ".borrow_mu", ".borrow_mut", "::new(", "\n    }", "ig", "igh", "eigh", "weigh", "weight", ");\r\n\r\n        "].iter().map(|x| { String::from(*x) }).collect()
}

fn most_common_fused_pair(v: Vec<String>) -> Option<(String, usize)> {
     if v.len() < 2 {
          return None;
     }
     let mut counts: HashMap<String, usize> = HashMap::new();
     // Iterate over adjacent pairs
     for pair in v.windows(2) {
          let fused = format!("{}{}", &pair[0], &pair[1]);
          let words = fused.split_whitespace().count();

          if words > 1 {
               continue;
          }

          // Increment count for this fused pair
          *counts.entry(fused).or_insert(0) += 1;
     }

     counts.into_iter()
         .max_by_key(|(pair, count)| {
              let len = pair.chars().count() as f64;
              (*count as f64 * len.sqrt()) as usize
         })
}

pub fn make_vocab(text: &str, token_num: usize) -> Vec<String> {
     let mut vocab: Vec<String> = vec!["<UNKNOWN>".to_string()];
     let mut chars = Vec::new();
     for x in text.chars() {
          let y = x.to_string();
          if !chars.contains(&y) {
               chars.push(y)
          }
     }
     let len = chars.len()+1;
     vocab.extend(chars);

     while vocab.len() < len+token_num {
          let split_text: Vec<String> = tokenize(text, &vocab).iter().map(|x1| {vocab[*x1].clone()}).collect();
          let Some((new_tok, count)) = most_common_fused_pair(split_text.clone()) else {
               println!("Not enough text to make {token_num} tokens!");
               break
          };

          if count < 2 {
               break
          }

          vocab.push(new_tok)
     }
     vocab
}
