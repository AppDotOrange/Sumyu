use std::collections::{BinaryHeap, HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use crate::vocabs;

pub fn xor() -> Vec<(Vec<f32>, Vec<f32>)> {
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
     ["<UNKNOWN>", "u", "s", "e", " ", "c", "r", "a", "t", ":", "f", "n", "_", "l", "m", "L", "M", ";", "\r", "\n", "d", "i", "o", "p", "b", "C", "h", "F", "N", "{", "}", "w", "(", ")", "-", ">", "S", "&", ",", "x", "g", "z", "=", "\"", "<", "U", "E", "R", ".", "*", "O", "T", "B", "!", "D", "v", "V", "0", "1", "/", "6", "4", "q", "|", "[", "]", "y", "H", "P", "k", "+", "2", "G", "j", "I", "3", "?", "\\", "Ж", "Х", "A", "J", "K", "Q", "W", "X", "Y", "Z", "5", "7", "8", "9", "'", "#", "$", "%", "^", "  ", "    ", "        ", "\n        ", "\r\n        ", " \"", "\",", "\n    ", "\n                ", "\r\n    ", "\r\n            ", "in", "t ", "en", "()", "er", " {", "se", "or", "at", "le", "let ", "= ", "pu", "\r\n     \"", "sel", "self", "on", "\r\n", " = ", ";\r\n        ", "\n                    ", "\",\r\n        ", "self.", ": ", "ra", " {\r\n            ", "ec", ");\r\n        ", ", ", "::", "re", "ata", "data", "ens", "ensor", "Tensor", " {\r\n        ", "\n            ", "iz", "ize", "size", "Vec", "Vec<", "ch", "ing", "mu", "mut ", "usize", "pub", "pub ", "ter", "}\r\n", "();\r\n        ", "t_", "rad", "grad", "\r\n    pub ", "ou", "th", "n ", "fn ", "pre", "prev", "one", "\n                        ", "inpu", "ro", ",\r\n            ", "\n        let ", "ex", "().", "\r\n            .", " -", "\r\n    }\r\n", "lone", "ar", "ne", "ma", " ->", " -> ", "id", " \", ", "clone", "tr", ".clone", "\r\n\r\n        ", ";\r\n\r\n        ", "(&", "(&self", "to", "\",\r\n          ", "ts", "len", "tring", ".clone()", "iter", ";\n                    ", "inne", "inner", "new", "be", "orro", "orrow", "borrow", ".borrow", "par", "sh", ".iter", ".iter().", "String", "}\r\n        ", "ca", "/ ", "// ", " th", "oca", "um", "im", "for", "tex", "Vec<Tensor", "Vec<Tensor>", "ontex", "contex", "len()", ".len()", ");\r\n            ", "ocab", "vocab", "ding", "dding", "bedding", "mbedding", "embedding", " {\r\n                ", ".borrow_", ".borrow_mu", ".borrow_mut", "vec", "\n                            ", "grad ", "s ", "\n                }", "ion", "for ", "gh", "push", ".push", "para", "param", ".push(", "::new", "\n\n        ", ";\n\n        ", "64", "f32", ".0", ".borrow_mut().", ".borrow_mut().grad ", "prev[", "atch", "batch", "inner.clone()", " {\n                    ", "el", "co", "\r\n                ", "p(", "l\",", "dim", "\n    }", "yer", "ayer", "layer", "(&self, ", ",\r\n        ", "data()", "\n                }\n                ", "de", "].borrow_mut().grad ", "map(", "map(|", ".iter().map(|", "inputs", " in", " in ", "use", "n\",", "ct"].iter().map(|x| { String::from(*x) }).collect()
}

pub fn ml_200_tok_vocab_v2() -> Vec<String> {
     ["<UNKNOWN>", "u", "s", "e", " ", "r", "a", "n", "d", "_", "i", "t", ":", "{", "D", "b", "o", ",", "N", "m", "l", "}", ";", "\r", "\n", "z", "S", "c", "T", "#", "[", "v", "(", "C", ")", "]", "p", "E", "g", "V", "<", "f", "6", "4", ">", "w", "-", "=", "0", ".", "1", "/", "q", "|", "&", "*", "h", "y", "x", "H", "M", "L", "P", "3", "2", "k", "O", "+", "B", "G", "j", "U", "I", "!", "\"", "?", "Ж", "Х", "R", "A", "F", "^", "W", "Y", "7", "5", "\\", "Q", "%", "9", "  ", "    ", "        ", "\n        ", "\r\n        ", "\r\n            ", "\n                ", "in", "\r\n", "er", "t ", "\r\n                ", "()", "en", "\r\n    ", " {", "se", "= ", "or", "le", "let ", "sel", "self", "at", "pu", ": ", "self.", " = ", "ra", "\n                        ", "\n                    ", "on", "\r\n        let ", "::", ", ", "\r\n                    ", "ed", "iz", "ize", "size", "ens", "ensor", "Tensor", "dat", "data", "ing", "ch", "\n            ", "re", "rad", "grad", "ec", ",\r\n            ", " {\r\n            ", ");", "pub", "pub ", "\r\n    pub ", "\r\n\r\n    pub ", "mu", "mut ", "usize", "();", "ro", "\r\n    }", "it", "\n    ", "n ", "Vec", "put", "input", "\r\n        }", "er.", "s: ", "inn", "inner.", "pre", "Vec<", "prev", "fn ", "al", "ding", "edding", "bedding", "mbedding", "embedding", "ne", "\r\n                .", "\n        let ", " -", "one", "\r\n            .", "inner.prev", " {\r\n        ", "().", " ->", " -> ", "iter", "atch", "id", "st", "ma", "s.", "(&", "(&self", "lone", "ar", "\r\n\r\n        ", "len", "_size", "clone", "inner.prev[", "ex", "yer", "ayer", "layer", "par", ";\n                    ", "new", "um", "co", "nt", "batch", "lo", "av", "aved", "orro", "orrow", "borrow", "rain", " {\r\n                ", "for", "_len", ".borrow", "/ ", "// ", "::new", "cont", "contex", "context", "\n                            ", "para", "param", "di", "grad ", "();\r\n\r\n        ", "clone()", "for ", "\n                }", " {\n                    ", "s ", "el", "Vec<Tensor", "Vec<Tensor>", "context_len", ",\r\n    ", ",\r\n        ", "uro", "p(", ");\r\n            ", "ct", "iter()", "map(", "map(|", "och", "poch", "epoch", "dim", "to", "len()", "inner.clone()", "embedding_", "embedding_dim", "in ", " in ", "64", "f32", "\n\n        ", "lect", "llect", "collect", "sh", "\n                }\n                ", "Saved", "(&self)", "+= ", "\r\n            }", ";\r\n\r\n        ", "de", ");\r\n\r\n        ", "uron", "neuron", "::new(", "ig", "igh", "ight", "eight", "weight", "\n    }", ".borrow_"].iter().map(|x| { String::from(*x) }).collect()
}

pub fn ml_200_tok_vocab_v3() -> Vec<String> {
     ["<UNKNOWN>", "u", "s", "e", " ", "r", "a", "n", "d", "_", "i", "t", ":", "{", "D", "b", "o", ",", "N", "m", "l", "}", ";", "\r", "\n", "z", "S", "c", "T", "#", "[", "v", "(", "C", ")", "]", "p", "E", "g", "V", "<", "f", "6", "4", ">", "w", "-", "=", "0", ".", "1", "/", "q", "|", "&", "*", "h", "y", "x", "H", "M", "L", "P", "3", "2", "k", "O", "+", "B", "G", "j", "U", "I", "!", "\"", "?", "Ж", "Х", "R", "A", "F", "^", "W", "Y", "7", "5", "\\", "Q", "%", "9", "  ", "    ", "        ", "\n        ", "\r\n        ", "\r\n            ", "\n                ", "in", "\r\n", "er", "t ", "\r\n                ", "()", "en", "\r\n    ", " {", "se", "= ", "or", "le", "let ", "sel", "self", "at", "pu", ": ", "self.", " = ", "ra", "\n                        ", "\n                    ", "on", "\r\n        let ", "::", ", ", "\r\n                    ", "ed", "iz", "ize", "size", "ens", "ensor", "Tensor", "dat", "data", "ing", "ch", "re", "\n            ", "ec", ",\r\n            ", "rad", "grad", " {\r\n            ", ");", "pub", "pub ", "\r\n    pub ", "\r\n\r\n    pub ", "mu", "mut ", "usize", "();", "\r\n    }", "ro", "t_", "\n    ", "n ", "Vec", "\r\n        }", "er.", "it", "pre", "inn", "inner.", "s: ", "inpu", "Vec<", "fn ", "prev", "al", "ding", "edding", "bedding", "mbedding", "embedding", "ne", "\r\n                .", "\n        let ", " -", "one", "\r\n            .", "inner.prev", "().", " {\r\n        ", " ->", " -> ", "atch", "id", "iter", "ma", "(&", "(&self", "s.", "lone", "ar", "len", "\r\n\r\n        ", "clone", "inner.prev[", "ex", "yer", "ayer", "layer", ";\n                    ", "par", "st", "new", "um", "co", "nt", "lo", "batch", "av", "aved", "orro", "orrow", "borrow", "rain", "input", "_size", " {\r\n                ", ".borrow", "/ ", "// ", "for", "::new", "cont", "contex", "para", "param", "\n                            ", "();\r\n\r\n        ", "grad ", "di", "clone()", "for ", " {\n                    ", "\n                }", "s ", "el", ",\r\n    ", ",\r\n        ", "Vec<Tensor", "Vec<Tensor>", "p(", ");\r\n            ", "context_", "context_len", "uro", "iter()", "ct", "dim", "len()", "to", "och", "poch", "epoch", "map(", "map(|", "inner.clone()", "embedding_", "embedding_dim", "lect", "llect", "collect", "64", "f32", "\n\n        ", "in ", " in ", "\n                }\n                ", "sh", "(&self)", "\r\n            }", "+= ", "Saved", "de", ";\r\n\r\n        ", ".borrow_", ".borrow_mu", ".borrow_mut", "::new(", "\n    }", "ig", "igh", "eigh", "weigh", "weight", ");\r\n\r\n        "].iter().map(|x| { String::from(*x) }).collect()
}

pub fn ml_v4() -> Vec<String> { // 250 tokens
     ["<UNKNOWN>", "", "u", "s", "e", " ", "r", "a", "n", "d", "_", "i", "t", ":", "{", "D", "b", "o", ",", "N", "m", "l", "}", ";", "\r", "\n", "z", "S", "c", "T", "#", "[", "v", "(", "C", ")", "]", "p", "E", "g", "V", "<", "f", "6", "4", ">", "w", "-", "=", "0", ".", "1", "/", "q", "|", "&", "*", "h", "y", "x", "H", "M", "L", "P", "3", "2", "k", "O", "+", "B", "G", "j", "U", "I", "!", "\"", "?", "Ж", "Х", "R", "A", "F", "^", "W", "Y", "7", "5", "\\", "Q", "%", "9", "  ", "    ", "        ", "\n        ", "\r\n        ", "\r\n            ", "\n                ", "in", "\r\n", "er", "t ", "\r\n                ", "()", "en", "\r\n    ", " {", "se", "= ", "or", "le", "let ", "sel", "self", "at", "pu", ": ", "self.", " = ", "ra", "\n                        ", "\n                    ", "on", "\r\n        let ", "::", ", ", "\r\n                    ", "ed", "iz", "ize", "size", "ens", "ensor", "Tensor", "ata", "data", "ing", "ch", "\n            ", "re", "rad", "grad", "ec", ",\r\n            ", " {\r\n            ", ");", "pub", "pub ", "\r\n    pub ", "\r\n\r\n    pub ", "mu", "mut ", "usize", "();", "\r\n    }", "ro", "n ", "it", "\n    ", "Vec", "put", "input", "\r\n        }", "er.", "pre", "inn", "inner.", "s: ", "Vec<", "prev", "fn ", "al", "ding", "edding", "bedding", "mbedding", "embedding", "ne", "\n        let ", "\r\n                .", " -", "one", "\r\n            .", "inner.prev", " {\r\n        ", "().", " ->", " -> ", "id", "iter", "st", "atch", "ma", "(&", "(&self", "s.", "lone", "ar", "\r\n\r\n        ", "len", "_size", "clone", "inner.prev[", "ex", ";\n                    ", "yer", "ayer", "layer", "par", "new", "co", "nt", "um", "lo", "batch", "av", "aved", "rain", "orro", "orrow", "borrow", " {\r\n                ", "/ ", "// ", "_len", "for", ".borrow", "::new", "cont", "contex", "context", "para", "param", "\n                            ", "();\r\n\r\n        ", "grad ", "di", "clone()", "for ", "\n                }", " {\n                    ", "context_len", "el", ",\r\n    ", ",\r\n        ", "s ", "Vec<Tensor", "Vec<Tensor>", "p(", ");\r\n            ", "uro", "iter()", "ct", "och", "poch", "epoch", "map(", "map(|", "to", "len()", "dim", "inner.clone()", "f6", "f32", "lect", "llect", "collect", "in ", " in ", "\n\n        ", "embedding_", "embedding_dim", "sh", "\n                }\n                ", "+= ", "(&self)", "Saved", "\r\n            }", ";\r\n\r\n        ", "de", ");\r\n\r\n        ", ".borrow_", ".borrow_mu", ".borrow_mut", "uron", "neuron", "batch_size", "\n    }", "ig", "igh", "ight", "eight", "weight", "::new(", "iter().", "max", ".borrow_mut().", ".borrow_mut().grad ", "].borrow_mut().grad ", "et", "push", "push(", "0.", "sum", "ou", "alize", "ialize", "collect()", "vec", "f ", "if ", "out", "output", "(&self, ", " =", ",\n        ", ",\n                ", "\n    }\n", "\n    }\n\n    ", ".data", "\r\n            let ", ");\r\n        ", "Embedding", "ca", "s,\r\n            ", "Train", ");\r\n                ", " {\r\n                    ", "ate", "str", "get", "arget", "target", "train", "saved", "self.batch_size", "\r\n                .collect()", "ion", "Op", "Op::", ".len()"].iter().map(|x| { String::from(*x) }).collect()
}

pub fn poke_v1() -> Vec<String> { // 250 tokens
    ["<UNKNOWN>", "", "N", "A", "M", "E", ":", " ", "B", "u", "l", "b", "a", "s", "r", "\n", "T", "Y", "P", "G", "o", "i", "n", "I", "L", "O", "v", "e", "g", "w", ",", "C", "h", "p", "y", "R", "W", "t", "d", "c", "k", ".", "x", "S", "m", "V", "f", "F", "z", "é", "4", "6", "0", "q", "D", "j", "’", "K", "2", "H", "U", "-", "1", "—", "♀", "♂", "'", "J", "Z", "8", ";", "Q", "!", "3", "X", "”", "“", "9", "5", "−", "7", "s ", "e ", ": ", "in", "th", "E: ", "t ", ", ", "er", "TY", "on", "d ", "an", "PE: ", "TYPE: ", "\nTYPE: ", "ME: ", "AME: ", "NAME: ", "\nNAME: ", "\n\nNAME: ", "ing", "TY: ", "ITY: ", "LITY: ", "ILITY: ", "BILITY: ", "ABILITY: ", "\nABILITY: ", "ar", "it", "en", "or", "y ", "the ", "at", "its ", "ing ", "es ", "\nE", "\nEN", "\nENT", "\nENTR", "\nENTRY", "\nENTRY:", "\nENTRY:\n", "ou", "on ", "o ", "al", "re", "to ", "is ", ". ", "st", "er ", "ic", "f ", "el", "and ", "it ", "of ", "il", "a ", "in ", "ed ", "h ", "Po", "ro", "ra", "le", "Pok", "Poké", "Pokém", "Pokémon ", "at ", "Th", "is", "ch", "en ", "ig", "ow", "un", "l ", "as ", "It ", "an ", "that ", "k ", "igh", "es", "ts ", "om", "ly ", "ea", "ith ", "with ", "im", "ur", "e, ", "ec", "ir", "this ", "em", "la", "for", "’s ", "ol", "us", "iv", "ight", "n ", "ta", "y, ", "sh", "od", "m ", "ev", "oun", "s, ", "The ", "wh", "Wat", "op", "ee", "ill ", "ug", "rom ", "from ", "This ", "le ", "al ", "sp", "are ", "id", "oc", "tr", "ac", "ion", "will ", "ap", "ma", "orma", "Norma", "can ", "ear", "Pokémon", "ss", "rass", "Grass", "ver", "tric", "the", "li", "ce ", "ds ", "ut", "s. ", "ve ", "Water ", "chic", "ychic", "sychic", "Psychic", "lo", "ng", "ent ", "et", "ly", "lying", "Flying", "ent", "up", "mor", "Pois", "St", "d, ", "by ", "roun", "Its ", "er, ", "ost", "for ", "pre", "wa", "ter", " of ", "se ", "air", "s.", "out ", "ke ", "Water", "bod", "t, ", "Normal", "ag", "has ", "ul", "ab", "ies ", "ound ", "if", "rag", "ves ", "their", "their ", "am", "Sh", "ighting", "Fighting", "ous ", "low", "ce", "con", "pro", "po", "ener", "een ", "hea", "body ", "ter ", "ang", "Drag", "pow", "Roc", "In", "into ", "ther", " Flying", "Fir", "ip", "row", "Stee", "Wh", "When ", "ctric", "lectric", "Electric", "one ", "oug", "ough ", "Poison ", "over", "Bug", " to ", "ed", "Sw", "ck", "Groun"].iter().map(|x| { String::from(*x) }).collect()
}

pub fn poke_v2() -> Vec<String> { // 300 tokens
    ["", "", "N", "A", "M", "E", ":", " ", "B", "u", "l", "b", "a", "s", "r", "\n", "T", "Y", "P", "G", "o", "i", "n", "I", "L", "O", "v", "e", "g", "w", ",", "C", "h", "p", "y", "R", "W", "t", "d", "c", "k", ".", "x", "S", "m", "V", "f", "F", "z", "é", "4", "6", "0", "q", "D", "j", "’", "K", "2", "H", "U", "-", "1", "—", "♀", "♂", "'", "J", "Z", "8", ";", "Q", "!", "3", "X", "”", "“", "9", "5", "−", "7", "s ", "e ", ": ", "th", "in", "E: ", "t ", ", ", "er", "TY", "on", "d ", "an", "ME: ", "AME: ", "NAME: ", "\nNAME: ", "\n\nNAME: ", "PE: ", "TYPE: ", "\nTYPE: ", "ing", "TY: ", "ITY: ", "LITY: ", "ILITY: ", "BILITY: ", "ABILITY: ", "\nABILITY: ", "ar", "it", "en", "or", "y ", "the ", "at", "its ", "ing ", "es ", "\nE", "\nEN", "\nENT", "\nENTR", "\nENTRY", "\nENTRY:", "\nENTRY:\n", "ou", "on ", "o ", "al", "re", "to ", "is ", ". ", "st", "er ", "ic", "f ", "el", "and ", "it ", "of ", "il", "a ", "in ", "ed ", "h ", "Po", "ro", "ra", "le", "Pok", "Poké", "Pokém", "Pokémon ", "at ", "Th", "is", "ch", "en ", "ig", "ow", "un", "l ", "as ", "It ", "an ", "that ", "k ", "igh", "ts ", "es", "ly ", "om", "ea", "ith ", "with ", "im", "ur", "e, ", "ec", "ir", "this ", "em", "la", "for", "’s ", "ol", "us", "iv", "ight", "n ", "ta", "y, ", "sh", "od", "m ", "ev", "oun", "s, ", "The ", "wh", "Wat", "op", "ee", "ill ", "ug", "rom ", "from ", "This ", "al ", "le ", "sp", "are ", "id", "oc", "tr", "ac", "ion", "will ", "ap", "can ", "ma", "orma", "Norma", "ear", "Pokémon", "ss", "rass", "Grass", "ver", "tric", "the", "li", "ce ", "ds ", "ut", "s. ", "ve ", "Water ", "chic", "ychic", "sychic", "Psychic", "lo", "ng", "ent ", "et", "ly", "lying", "Flying", "ent", "up", "mor", "Pois", "St", "d, ", "by ", "Its ", "roun", "er, ", "ost", "for ", "pre", "wa", "ter", " of ", "se ", "air", "s.", "out ", "ke ", "Water", "t, ", "bod", "Normal", "ag", "has ", "ab", "ul", "ies ", "ound ", "if", "ves ", "rag", "their", "their ", "am", "Sh", "Fight", "Fighting", "ous ", "low", "ce", "pro", "con", "po", "body ", "ter ", "een ", "ener", "hea", "ang", "Drag", "In", "pow", "Roc", "into ", "Fir", " Flying", "ther", "ip", "Wh", "When ", "Stee", "row", "ctric", "lectric", "Electric", "one ", "oug", "ough ", "Poison ", "over", "Bug", " to ", "ed", "Sw", "Groun", "ck", "tim", "tle", "tion", " and ", "mo", "eg", "ell ", "them ", " the ", "be ", "host", "Ghost", "own ", "per", "ng ", "ick", "Dar", "ave ", "Dragon", "Fair", "ex", "e.", "so ", "e. ", "ers ", "attle", "when ", "Grass ", "roug", "har", "com", "ectric", "electric", "Poison", "reat", "get", "ch ", "rough ", "ad", "have ", "of", "Fairy", "like ", "other", "through ", "kes ", "um", "y. ", "Pokémon’s ", "grow"].iter().map(|x| { String::from(*x) }).collect()
}

pub fn tale_v1() -> Vec<String> { // 250 tokens
    ["<UNKNOWN>", "", "T", "H", "E", " ", "F", "R", "O", "G", "-", "K", "I", "N", ";", ",", "Y", "\n", "n", "o", "l", "d", "t", "i", "m", "e", "s", "w", "h", "g", "a", "v", "r", "u", "b", "f", "y", "c", ".", "C", "’", "k", "W", "p", "A", "D", "“", "?", "”", "S", "!", "B", "q", "j", "_", "x", ":", "M", "L", "[", "]", "V", "U", "P", "Z", "(", "z", ")", "‘", "J", "Q", "<", ">", "X", "e ", "th", "d ", " th", " a", "er", " the ", "t ", "nd ", "in", " and ", ", ", "s ", "and ", "ou", "en", "he ", "to", "the ", "ed ", "ing", "\n\n", " h", "on", "an", " s", "or", "ll", "ea", " w", "as ", ". ", "at ", "y ", "you", "  ", "hi", "e, ", "ai", "no", "d, ", "ther", "hen", ".\n\n", "”\n\n", "for", "re", "of", "ha", "it", "om", "her", "to ", "st", "the", "ow", "gh", "ll ", "ut ", "was ", ".”\n\n", "ld ", "se", "ing ", "le ", "ver", "ve ", "ch", "ith", " the", "be", "not ", "is ", "ca", "The ", " to ", "she ", " m", "le", "oo", "wa", "had ", "s, ", "    ", " you", "ill ", "with", "aid ", "me ", "ter", "that ", "ck", "ri", "d\n", "sh", "la", "out ", "ti", " was ", "ould ", "ir", " in", "other", " ha", " she ", "li", "him", "a ", "un", "tle ", "ear", "but ", "one ", "t, ", "ght ", "ever", " her", "ly ", "ent ", "ar", "go", "ain", "aid, ", "Then", " he ", ",\n", "when", " it", " said ", "his ", "e\n", "ttle ", "little ", "came ", "es ", "hat ", "ho", " f", " d", "y, ", "ed, ", "rea", "they ", "ng", "said, ", " his ", "all ", "wer", " of", "sel", " be", "ra", " b", "ld", "up", "ke ", "went ", "ust", "But ", " that ", "And ", ",”", " will ", "When", "so", "ing, ", "self", "ce ", " had ", "ed", "own", "gain", "hich", "too", "if", "who", "rou", " they ", "thing ", "est", "com", "ro", "’s ", "here ", " with", "id ", "have ", " p", "sa", "      ", "now", "ven", "and\n", "es", "ked ", "into ", "were ", "ep", "ta", "t\n", "et ", " for", "el", "ell", "fu", "man", "e. ", "ong", "im", "der", "“I", " on", "long", "si", " a ", "ted ", "s\n", "da", " him", "ill", " as ", "could ", "took", "ur", "ne", "rom", "the\n", "King ", "de ", ",\n      ", "them", "pp", " an", "mor", "over", "which", "al", "oman", "ish", "King", "She ", "will ", " them", "did ", " into ", "oor", " have ", " the\n", "ghter", "ughter"].iter().map(|x| { String::from(*x) }).collect()
}

pub fn recipe_v1() -> Vec<String> { // 250 tokens
    ["", "", "A", "i", "r", " ", "F", "y", "e", "P", "o", "t", "a", "S", "l", "c", "s", "w", "h", "D", "p", "n", "g", "u", "\n", ":", "T", "f", ",", "v", "d", "b", "k", "m", ".", "—", "'", "I", "3", "/", "4", "1", "2", "W", "C", "B", "5", "R", "0", "(", ")", "-", "G", "7", "z", "j", "8", "9", ";", "6", "M", "E", "?", "!", "q", "¼", "O", "x", "H", "®", "\"", "*", "L", "U", "Q", "N", "½", "Y", "K", "J", "¾", "ñ", "–", "™", "V", "é", "⅓", "°", "⅛", " t", " a", "in", "er", "e ", "es", "on", "d ", "re", ". ", " s", " c", "nd ", " and ", "ea", " th", " the ", "ing", " f", "en", " to", " in", "po", "th", " b", "poon", " m", "ic", "yer", "ryer", " fryer", "ir", "ed ", ".\n", "il", "eas", "easpoon", " teaspoon", "ut", "gre", "ar", " w", "at", " d", "\n1", "ith", " o", "al", " with", " p", "grees", "egrees", " degrees", " degrees ", "et", "an", "ion", "ri", "ou", "ch", "la", "tion", "or", "ow", "st", "ick", " min", " minut", " minutes", " co", "\n\n", " ch", "icken", "ent", "per", " to ", "ok", "is", "ve ", "pper", "epper", "air", "eat", "to", ", ", "and ", "\n\nD", "ed", "ab", "un", "espoon", "lespoon", "ablespoon", " tablespoon", "lic", "rea", "ce ", "til", "until", " until", "co", "ay", "heat", "reheat", " teaspoon ", "li", "sh", ".\n\n", "ction", "ctions", " chicken", " air", "e, ", "el", "alt", "--", "der", "ents", "ients", "dients", "gredients", "ngredients", "ngredients:", "Ingredients:", "gar", "garlic", " of", " (", "ections", "irections", "\n\nDirections", "\n\nDirections:", "le ", "up", " cup", "lace ", "as", "ption", "ription", "cription", "escription", "\n\nDescription", "\n\nDescription:", "\n\nDescription:\n", "\n1/", " for", " tablespoons", " minutes. ", "pepper", "ket", "asket", "Preheat", " sp", "our", " cook", " 1", "ly", "Place ", " on", " an", "ru", "salt", "ver", "out", "oil", "eason", "Air", " basket", "owl", "ter", "em", "with", "Fryer", " Fryer", " Fryer ", " h", "ro", "id", "\n1. ", "gh", " pepper", "the ", "C)", "ato", "tato", " bowl", "ray", " spray", "ge ", "au", " sau", "ure", "Ingredients:\n", "read ", "live ", "cook", "ak", "ture", "tatoes", "utter", " teaspoons", "\n\nAir", " e", "2. ", "---", "owder", " powder", "ees", " chees", "3. ", " into", "all", "00", "s, ", " re", "erve ", "ss", " cooking", "ran", "\n2", "get", " olive ", "her", "chicken", "Co", "ender", " sauc", "ted ", "ix", "ixture", "sp", "mb", " season", "ey", "about", "ra", "ces", "4. ", " over", "ck", "ll", " g", " l", "igh", "it", "Th", "e. ", " are", " mixture", "ry"].iter().map(|x| { String::from(*x) }).collect()
}

use std::fs::File;
use std::io::{BufWriter, Write};

pub fn dump_vocab_as_rust(
    vocab: &[String],
    path: &str,
) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "vec![")?;

    for token in vocab {
        // {:?} produces a valid escaped Rust string literal.
        writeln!(w, "    {:?},", token)?;
    }

    writeln!(w, "]")?;

    Ok(())
}

const NONE: u32 = u32::MAX;
const REMOVED: u32 = u32::MAX - 1;

// ------------------------------------------------------------
// Fast integer hasher
//
// The input here is trusted/local data, so we don't need
// HashMap's DoS-resistant SipHash. This is substantially faster
// for integer-heavy tables.
//
// ------------------------------------------------------------

#[derive(Default)]
struct FastHasher(u64);

impl FastHasher {
    #[inline]
    fn mix(mut x: u64) -> u64 {
        x ^= x >> 30;
        x = x.wrapping_mul(0xbf58476d1ce4e5b9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }
}

impl Hasher for FastHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // Fallback for things that don't use write_uXX().
        let mut h = 0x9e3779b97f4a7c15u64;

        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }

        self.0 = Self::mix(h);
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.0 = Self::mix(value);
    }

    #[inline]
    fn write_u32(&mut self, value: u32) {
        self.0 = Self::mix(value as u64);
    }

    #[inline]
    fn write_usize(&mut self, value: usize) {
        self.0 = Self::mix(value as u64);
    }
}

type FastHashMap<K, V> =
HashMap<K, V, BuildHasherDefault<FastHasher>>;

type FastHashSet<T> =
HashSet<T, BuildHasherDefault<FastHasher>>;

// ------------------------------------------------------------
// Byte-aware trie tokenizer
//
// Vocabulary entries can be:
//   - normal UTF-8 strings: "hello", "é", " world"
//   - single byte fallback: "<0xE9>"
//   - byte-encoded learned tokens:
//         "<0xE2><0x82>"
//         "<0xC3><0xA9>"
//     (used when the merged byte sequence is not itself valid
//      UTF-8 yet)
//
// Special tokens such as "<EOT>" remain literal strings.
//
// The trie therefore always matches the decoded byte sequence,
// never the literal "<0xXX>" representation.
// ------------------------------------------------------------

#[inline]
fn hex_value(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Parse one exact `<0xXX>` token.
#[inline]
fn parse_byte_token(token: &str) -> Option<u8> {
    let b = token.as_bytes();

    if b.len() != 6 ||
        b[0] != b'<' ||
        b[1] != b'0' ||
        b[2] != b'x' ||
        b[5] != b'>'
    {
        return None;
    }

    let hi = hex_value(b[3])?;
    let lo = hex_value(b[4])?;

    Some((hi << 4) | lo)
}

/// Returns true if the entire token is made only from
/// `<0xXX>` byte markers.
///
/// This is how we represent learned byte sequences that are
/// not valid UTF-8 yet.
#[inline]
fn is_byte_encoded_token(token: &str) -> bool {
    if token.is_empty() {
        return false;
    }

    let bytes = token.as_bytes();
    let mut pos = 0;

    while pos < bytes.len() {
        if pos + 6 > bytes.len() {
            return false;
        }

        if parse_byte_token(
            unsafe { std::str::from_utf8_unchecked(&bytes[pos..pos + 6]) }
        ).is_none() {
            return false;
        }

        pos += 6;
    }

    true
}

/// Append the decoded byte representation of a vocabulary token.
#[inline]
fn append_token_bytes(token: &str, out: &mut Vec<u8>) {
    if is_byte_encoded_token(token) {
        let bytes = token.as_bytes();
        let mut pos = 0;

        while pos < bytes.len() {
            let byte =
                parse_byte_token(
                    unsafe {
                        std::str::from_utf8_unchecked(
                            &bytes[pos..pos + 6]
                        )
                    }
                )
                    .expect("validated byte token");

            out.push(byte);
            pos += 6;
        }
    } else {
        out.extend_from_slice(token.as_bytes());
    }
}

/// Append the decoded byte representation without allocating.
#[inline]
fn append_token_to_string(token: &str, out: &mut String) {
    if let Some(byte) = parse_byte_token(token) {
        // Fast path for the overwhelmingly common byte fallback case.
        let _ = byte;

        // This cannot directly be pushed as a char for arbitrary bytes,
        // so use the byte builder below instead.
        unreachable!();
    }

    out.push_str(token);
}

#[inline]
fn decoded_token_len(token: &str) -> usize {
    if is_byte_encoded_token(token) {
        token.len() / 6
    } else if let Some(_) = parse_byte_token(token) {
        1
    } else {
        token.len()
    }
}

struct TrieNode {
    children: FastHashMap<u8, u32>,
    id: u32,
}

pub(crate) struct Trie {
    nodes: Vec<TrieNode>,
    byte_fallback_base: u32,
}

impl Trie {
    pub(crate) fn from_vocab(vocab: &[String]) -> Self {
        let total_bytes: usize =
            vocab
                .iter()
                .map(|token| decoded_token_len(token))
                .sum();

        let mut nodes =
            Vec::with_capacity(total_bytes + 1);

        nodes.push(TrieNode {
            children: FastHashMap::default(),
            id: NONE,
        });

        for (id, token) in vocab.iter().enumerate() {
            let id = id as u32;

            let mut current = 0u32;

            if is_byte_encoded_token(token) {
                let bytes = token.as_bytes();
                let mut pos = 0;

                while pos < bytes.len() {
                    let byte =
                        parse_byte_token(
                            unsafe {
                                std::str::from_utf8_unchecked(
                                    &bytes[pos..pos + 6]
                                )
                            }
                        )
                            .expect("validated byte-encoded token");

                    let current_idx =
                        current as usize;

                    if let Some(&next) =
                        nodes[current_idx]
                            .children
                            .get(&byte)
                    {
                        current = next;
                    } else {
                        let next =
                            nodes.len() as u32;

                        nodes.push(TrieNode {
                            children:
                            FastHashMap::default(),
                            id: NONE,
                        });

                        nodes[current_idx]
                            .children
                            .insert(byte, next);

                        current = next;
                    }

                    pos += 6;
                }
            } else if let Some(byte) = parse_byte_token(token) {
                let current_idx =
                    current as usize;

                if let Some(&next) =
                    nodes[current_idx]
                        .children
                        .get(&byte)
                {
                    current = next;
                } else {
                    let next =
                        nodes.len() as u32;

                    nodes.push(TrieNode {
                        children:
                        FastHashMap::default(),
                        id: NONE,
                    });

                    nodes[current_idx]
                        .children
                        .insert(byte, next);

                    current = next;
                }
            } else {
                for &byte in token.as_bytes() {
                    let current_idx =
                        current as usize;

                    if let Some(&next) =
                        nodes[current_idx]
                            .children
                            .get(&byte)
                    {
                        current = next;
                    } else {
                        let next =
                            nodes.len() as u32;

                        nodes.push(TrieNode {
                            children:
                            FastHashMap::default(),
                            id: NONE,
                        });

                        nodes[current_idx]
                            .children
                            .insert(byte, next);

                        current = next;
                    }
                }
            }

            // Empty reserved tokens are not actual trie tokens.
            if !token.is_empty() {
                nodes[current as usize].id = id;
            }
        }

        let byte_fallback_base = vocab
            .iter()
            .position(|token| token == "<0x00>")
            .expect("Vocabulary is missing <0x00>") as u32;

        assert!(
            byte_fallback_base as usize + 256 <= vocab.len(),
            "Byte fallback block is incomplete"
        );

        #[cfg(debug_assertions)]
        for byte in 0u16..=255 {
            let id =
                byte_fallback_base as usize +
                    byte as usize;

            assert_eq!(
                vocab[id],
                format!("<0x{:02X}>", byte),
            );
        }

        Self {
            nodes,
            byte_fallback_base,
        }
    }

    #[inline]
    pub(crate) fn tokenize_u32(
        &self,
        text: &str,
    ) -> Vec<u32> {
        let bytes = text.as_bytes();

        let mut result =
            Vec::with_capacity(bytes.len());

        let mut pos = 0usize;

        while pos < bytes.len() {
            let mut current = 0u32;
            let mut best_id = NONE;
            let mut best_len = 0usize;

            let mut j = pos;

            while j < bytes.len() {
                let byte = bytes[j];

                let Some(&next) =
                    self.nodes[current as usize]
                        .children
                        .get(&byte)
                else {
                    break;
                };

                current = next;
                j += 1;

                let id =
                    self.nodes[current as usize].id;

                if id != NONE {
                    best_id = id;
                    best_len = j - pos;
                }
            }

            debug_assert!(
                best_id != NONE,
                "Byte fallback vocabulary should match every input byte"
            );

            if best_id == NONE {
                // Defensive fallback for malformed vocabularies.
                result.push(
                    self.byte_fallback_base +
                        bytes[pos] as u32,
                );
                pos += 1;
            } else {
                result.push(best_id);
                pos += best_len;
            }
        }

        result
    }
}

// ------------------------------------------------------------
// Token metadata
// ------------------------------------------------------------

#[derive(Clone, Copy)]
struct TokenMeta {
    byte_len: u32,
    word_count: u32,
    starts_non_ws: bool,
    ends_non_ws: bool,

    // Special/reserved tokens are barriers and must not be merged.
    mergeable: bool,
}

impl TokenMeta {
    fn from_token(
        token: &str,
        mergeable: bool,
    ) -> Self {
        // Normal UTF-8 token.
        if !is_byte_encoded_token(token) {
            let mut word_count = 0u32;
            let mut in_word = false;
            let mut starts_non_ws = false;
            let mut ends_non_ws = false;
            let mut first = true;

            for ch in token.chars() {
                if ch.is_whitespace() {
                    in_word = false;
                    ends_non_ws = false;
                } else {
                    if first {
                        starts_non_ws = true;
                    }

                    if !in_word {
                        word_count += 1;
                        in_word = true;
                    }

                    ends_non_ws = true;
                }

                first = false;
            }

            return Self {
                byte_len: token.len() as u32,
                word_count,
                starts_non_ws,
                ends_non_ws,
                mergeable,
            };
        }

        // Byte-encoded token.
        let bytes = token.as_bytes();

        let mut word_count = 0u32;
        let mut in_word = false;
        let mut starts_non_ws = false;
        let mut ends_non_ws = false;
        let mut first = true;

        let mut pos = 0usize;

        while pos < bytes.len() {
            let byte =
                parse_byte_token(
                    unsafe {
                        std::str::from_utf8_unchecked(
                            &bytes[pos..pos + 6]
                        )
                    }
                )
                    .expect("validated byte token");

            if byte.is_ascii_whitespace() {
                in_word = false;
                ends_non_ws = false;
            } else {
                if first {
                    starts_non_ws = true;
                }

                if !in_word {
                    word_count += 1;
                    in_word = true;
                }

                ends_non_ws = true;
            }

            first = false;
            pos += 6;
        }

        Self {
            byte_len: (bytes.len() / 6) as u32,
            word_count,
            starts_non_ws,
            ends_non_ws,
            mergeable,
        }
    }

    #[inline]
    fn fuse(a: Self, b: Self) -> Self {
        let merges_words =
            a.ends_non_ws &&
                b.starts_non_ws &&
                a.word_count > 0 &&
                b.word_count > 0;

        Self {
            byte_len:
            a.byte_len + b.byte_len,

            word_count:
            a.word_count +
                b.word_count -
                merges_words as u32,

            starts_non_ws:
            if a.byte_len != 0 {
                a.starts_non_ws
            } else {
                b.starts_non_ws
            },

            ends_non_ws:
            if b.byte_len != 0 {
                b.ends_non_ws
            } else {
                a.ends_non_ws
            },

            mergeable:
            a.mergeable &&
                b.mergeable,
        }
    }

    #[inline]
    fn valid_fusion(
        a: Self,
        b: Self,
    ) -> bool {
        a.mergeable &&
            b.mergeable &&
            Self::fuse(a, b).word_count <= 1
    }
}

// ------------------------------------------------------------
// Pair helpers
// ------------------------------------------------------------

#[inline]
fn pair_key(a: u32, b: u32) -> u64 {
    ((a as u64) << 32) | (b as u64)
}

#[inline]
fn pair_ids(key: u64) -> (u32, u32) {
    (
        (key >> 32) as u32,
        key as u32,
    )
}

#[inline]
fn pair_score(
    key: u64,
    count: u32,
    meta: &[TokenMeta],
) -> u64 {
    let (a, b) = pair_ids(key);

    let len =
        meta[a as usize].byte_len +
            meta[b as usize].byte_len;

    // score = count * sqrt(len)
    //
    // sqrt is monotonic, but we still want length to matter.
    // Keeping the calculation in f64 is fine here because this
    // happens only when heap entries are refreshed, not once per
    // token occurrence.
    ((count as f64) * (len as f64).sqrt()) as u64
}

// ------------------------------------------------------------
// Pair index
//
// counts:
//     current number of occurrences of each pair
//
// occurrences:
//     node IDs that have represented that pair
//
// heap:
//     candidates for the highest-scoring pair
//
// The occurrence vectors deliberately allow stale entries.
// When a pair is selected, they are validated against the
// current linked sequence. This avoids expensive deletion from
// occurrence sets on every local change.
// ------------------------------------------------------------

struct PairIndex {
    counts: FastHashMap<u64, u32>,

    occurrences:
        FastHashMap<u64, Vec<u32>>,

    heap:
        BinaryHeap<(u64, u64, u32)>,

    dirty:
        FastHashSet<u64>,
}

impl PairIndex {
    fn build(
        tokens: &[u32],
        meta: &[TokenMeta],
    ) -> Self {
        let mut counts =
            FastHashMap::default();

        let mut occurrences =
            FastHashMap::default();

        if tokens.len() >= 2 {
            for i in 0..tokens.len() - 1 {
                let a = tokens[i] as usize;
                let b = tokens[i + 1] as usize;

                let ma = meta[a];
                let mb = meta[b];

                if !TokenMeta::valid_fusion(ma, mb) {
                    continue;
                }

                let key =
                    pair_key(
                        tokens[i],
                        tokens[i + 1],
                    );

                *counts
                    .entry(key)
                    .or_insert(0) += 1;

                occurrences
                    .entry(key)
                    .or_insert_with(Vec::new)
                    .push(i as u32);
            }
        }

        let mut index = Self {
            counts,
            occurrences,
            heap: BinaryHeap::new(),
            dirty: FastHashSet::default(),
        };

        for (&key, &count) in index.counts.iter() {
            if count >= 2 {
                index.heap.push((
                    pair_score(key, count, meta),
                    key,
                    count,
                ));
            }
        }

        index
    }

    #[inline]
    fn add_edge(
        &mut self,
        node: u32,
        a: u32,
        b: u32,
        meta: &[TokenMeta],
    ) {
        let ma = meta[a as usize];
        let mb = meta[b as usize];

        if !TokenMeta::valid_fusion(ma, mb) {
            return;
        }

        let key = pair_key(a, b);

        *self.counts.entry(key).or_insert(0) += 1;

        self.occurrences
            .entry(key)
            .or_insert_with(Vec::new)
            .push(node);

        self.dirty.insert(key);
    }

    #[inline]
    fn remove_edge(
        &mut self,
        a: u32,
        b: u32,
        meta: &[TokenMeta],
    ) {
        let ma = meta[a as usize];
        let mb = meta[b as usize];

        if !TokenMeta::valid_fusion(ma, mb) {
            return;
        }

        let key = pair_key(a, b);

        let Some(count) = self.counts.get_mut(&key)
        else {
            return;
        };

        *count -= 1;

        if *count == 0 {
            self.counts.remove(&key);

            // The old occurrence vector is now useless.
            self.occurrences.remove(&key);
        }

        self.dirty.insert(key);
    }

    fn flush_heap(&mut self, meta: &[TokenMeta]) {
        for key in self.dirty.drain() {
            if let Some(&count) = self.counts.get(&key) {
                if count >= 2 {
                    self.heap.push((
                        pair_score(key, count, meta),
                        key,
                        count,
                    ));
                }
            }
        }
    }

    fn best_pair(
        &mut self,
        _meta: &[TokenMeta],
    ) -> Option<(u64, u32)> {
        while let Some((
                           _score,
                           key,
                           snapshot_count,
                       )) = self.heap.pop()
        {
            match self.counts.get(&key) {
                Some(&current_count)
                if current_count == snapshot_count &&
                    current_count >= 2 =>
                    {
                        return Some((key, current_count));
                    }

                _ => {
                    // Stale heap entry.
                }
            }
        }

        None
    }
}

// ------------------------------------------------------------
// Incremental pair merging
//
// Sequence is represented as a doubly linked list using u32s.
// We never physically remove elements from the Vec.
//
// This is dramatically cheaper than rebuilding the whole
// token vector after every vocabulary addition.
// ------------------------------------------------------------

fn merge_pair(
    key: u64,
    new_id: u32,

    tokens: &mut [u32],
    prev: &mut [u32],
    next: &mut [u32],

    pair_index: &mut PairIndex,
    meta: &[TokenMeta],
) -> usize {
    // Take the candidate occurrences out of the index.
    //
    // They may contain stale entries, so they are validated below.
    let mut candidates =
        pair_index
            .occurrences
            .remove(&key)
            .unwrap_or_default();

    // Node IDs preserve original text order, because merges only
    // remove the right-hand node of a pair.
    candidates.sort_unstable();

    let (wanted_a, wanted_b) =
        pair_ids(key);

    let mut merged = 0usize;

    for cur_u32 in candidates {
        let cur = cur_u32 as usize;

        // Node was already removed by an overlapping merge.
        if next[cur] == REMOVED {
            continue;
        }

        let right_u32 = next[cur];

        if right_u32 == NONE {
            continue;
        }

        let right = right_u32 as usize;

        // It may no longer be the same pair because a previous
        // merge modified this region.
        if tokens[cur] != wanted_a ||
            tokens[right] != wanted_b
        {
            continue;
        }

        let left_u32 = prev[cur];
        let after_u32 = next[right];

        // --------------------------------------------
        // Remove old edges:
        //
        // left -> A
        // A    -> B
        // B    -> after
        // --------------------------------------------

        if left_u32 != NONE {
            let left = left_u32 as usize;

            pair_index.remove_edge(
                tokens[left],
                tokens[cur],
                meta,
            );
        }

        pair_index.remove_edge(
            tokens[cur],
            tokens[right],
            meta,
        );

        if after_u32 != NONE {
            let after = after_u32 as usize;

            pair_index.remove_edge(
                tokens[right],
                tokens[after],
                meta,
            );
        }

        // --------------------------------------------
        // Merge A+B into current node.
        // --------------------------------------------

        tokens[cur] = new_id;
        next[cur] = after_u32;

        if after_u32 != NONE {
            prev[after_u32 as usize] = cur_u32;
        }

        // Mark the right node as dead.
        next[right] = REMOVED;
        prev[right] = REMOVED;

        // --------------------------------------------
        // Add new edges:
        //
        // left -> NEW
        // NEW  -> after
        // --------------------------------------------

        if left_u32 != NONE {
            let left = left_u32 as usize;

            pair_index.add_edge(
                left_u32,
                tokens[left],
                new_id,
                meta,
            );
        }

        if after_u32 != NONE {
            let after = after_u32 as usize;

            pair_index.add_edge(
                cur_u32,
                new_id,
                tokens[after],
                meta,
            );
        }

        merged += 1;
    }

    // Update the heap once after the entire merge.
    pair_index.flush_heap(meta);

    merged
}

// ------------------------------------------------------------
// Fully optimized vocab builder
// ------------------------------------------------------------

fn add_byte_fallback_tokens(vocab: &mut Vec<String>) {
    for byte in 0u16..=255 {
        vocab.push(format!("<0x{:02X}>", byte));
    }
}

pub fn make_vocab(
    text: &str,
    token_num: usize,
    reserved_token_num: usize,
    special_tokens: Option<&[&str]>,
) -> Vec<String> {
    let now = std::time::Instant::now();
    // --------------------------------------------------------
    // Initial vocabulary:
    //   0..=reserved_token_num-1 -> reserved empty IDs
    //   reserved_token_num      -> extra empty slot
    //   special tokens
    //   256 byte fallback tokens
    // --------------------------------------------------------

    let special_count =
        special_tokens.map_or(0, |s| s.len());

    let initial_vocab_size =
        reserved_token_num +
            1 +
            special_count +
            256;

    let mut vocab =
        Vec::with_capacity(
            initial_vocab_size + token_num,
        );

    // Important:
    // reserved_token_num = 1
    // => two empty slots: IDs 0 and 1.
    vocab.resize(
        reserved_token_num + 1,
        String::new(),
    );

    // The special-token IDs start AFTER the entire reserved prefix.
    let special_start =
        vocab.len();

    if let Some(special) = special_tokens {
        vocab.extend(
            special.iter().map(|token| (*token).to_string())
        );
    }

    let byte_fallback_base =
        vocab.len();

    add_byte_fallback_tokens(&mut vocab);

    debug_assert_eq!(
        byte_fallback_base + 256,
        vocab.len()
    );

    // `token_num` is the number of NEW learned tokens.
    let target_vocab_size =
        vocab.len() + token_num;

    if vocab.len() > u32::MAX as usize {
        panic!("Vocabulary exceeds u32 token-ID capacity");
    }

    // --------------------------------------------------------
    // Token metadata
    //
    // Reserved slots are not mergeable.
    // Special tokens are not mergeable.
    // Byte fallback tokens are mergeable.
    // --------------------------------------------------------

    let mut meta =
        Vec::with_capacity(target_vocab_size);

    for id in 0..vocab.len() {
        let mergeable =
            id >= byte_fallback_base;

        meta.push(
            TokenMeta::from_token(
                &vocab[id],
                mergeable,
            )
        );
    }

    // --------------------------------------------------------
    // Initial tokenization
    //
    // The base vocabulary already contains every byte,
    // so the trie can tokenize the whole corpus directly.
    //
    // Special tokens automatically win through longest-match.
    // --------------------------------------------------------

    let trie =
        Trie::from_vocab(&vocab);

    let mut tokens =
        trie.tokenize_u32(text);

    if tokens.is_empty() {
        return vocab;
    }

    // --------------------------------------------------------
    // Convert initial tokenization into a linked sequence.
    // --------------------------------------------------------

    let n =
        tokens.len();

    let mut prev =
        vec![NONE; n];

    let mut next =
        vec![NONE; n];

    for i in 0..n {
        if i > 0 {
            prev[i] =
                (i - 1) as u32;
        }

        if i + 1 < n {
            next[i] =
                (i + 1) as u32;
        }
    }

    // --------------------------------------------------------
    // Build initial pair counts + occurrence index.
    // --------------------------------------------------------

    let mut pair_index =
        PairIndex::build(
            &tokens,
            &meta,
        );

    // --------------------------------------------------------
    // Incremental vocabulary construction.
    // --------------------------------------------------------

    while vocab.len() < target_vocab_size {
        let Some((best_key, count)) =
            pair_index.best_pair(&meta)
        else {
            println!(
                "Not enough text to make {} tokens!",
                token_num
            );
            break;
        };

        if count < 2 {
            break;
        }

        let (a, b) =
            pair_ids(best_key);

        // ----------------------------------------------------
        // Construct the new token exactly once.
        // ----------------------------------------------------

        // --------------------------------------------------------
        // Construct the new token from DECODED bytes.
        //
        // This is crucial:
        //
        //   "<0xC3>" + "<0xA9>"
        //
        // must become:
        //
        //   "é"
        //
        // rather than:
        //
        //   "<0xC3><0xA9>"
        //
        // For incomplete/invalid UTF-8 byte sequences, keep the
        // canonical `<0xXX>` representation.
        // --------------------------------------------------------

        let a_token =
            &vocab[a as usize];

        let b_token =
            &vocab[b as usize];

        let decoded_len =
            meta[a as usize].byte_len as usize +
                meta[b as usize].byte_len as usize;

        let mut bytes =
            Vec::with_capacity(decoded_len);

        append_token_bytes(
            a_token,
            &mut bytes,
        );

        append_token_bytes(
            b_token,
            &mut bytes,
        );

        let new_token =
            match String::from_utf8(bytes.clone()) {
                Ok(s) => s,

                Err(_) => {
                    let mut encoded =
                        String::with_capacity(
                            bytes.len() * 6
                        );

                    for byte in bytes {
                        use std::fmt::Write;

                        write!(
                            encoded,
                            "<0x{:02X}>",
                            byte
                        )
                            .unwrap();
                    }

                    encoded
                }
            };

        // ----------------------------------------------------
        // Metadata.
        // ----------------------------------------------------

        let new_meta =
            TokenMeta::fuse(
                meta[a as usize],
                meta[b as usize],
            );

        let new_id =
            vocab.len() as u32;

        vocab.push(new_token);
        meta.push(new_meta);

        // ----------------------------------------------------
        // Merge only actual occurrences.
        // ----------------------------------------------------

        let merged =
            merge_pair(
                best_key,
                new_id,

                &mut tokens,
                &mut prev,
                &mut next,

                &mut pair_index,
                &meta,
            );

        debug_assert!(
            merged > 0,
            "Best pair existed in the count table but had no live occurrences"
        );

        if merged == 0 {
            break;
        }
    }
    println!("Vocabulary creation finished in {:?}.", now.elapsed());

    vocab
}

pub struct Config<'a> {
     pub lr: f32,
     pub batch_size: usize,
     pub max_batches_per_epoch: usize, // 0 means no limit
     pub vocab: Vec<String>,
     pub context_len: usize,
     pub emb_dim: usize,
     pub hidden_dim: &'a [usize],
     pub epochs: usize,
}

impl<'a> Config<'a> {
    #[allow(clippy::too_many_arguments)]
     pub fn new(lr: f32,
                batch_size: usize,
                max_batches_per_epoch: usize,
                vocab: Vec<String>,
                context_len: usize,
                emb_dim: usize,
                hidden_dim: &'a [usize],
                epochs: usize,) -> Self {
          Self {
               lr,
               batch_size,
               max_batches_per_epoch,
               vocab,
               context_len,
               emb_dim,
               hidden_dim,
               epochs,
          }
     }
}

pub fn minimodel_config() -> Config<'static> {
     Config::new(0.01,
          32,
          10,
          ml_200_tok_vocab_v3(),
          8,
          20,
          &[200],
          500,
     )
}

pub fn rustception_optimized() -> Config<'static> {
     Config::new(0.01,
                 32,
                 0, // no limit
                 ml_200_tok_vocab_v3(),
                 32,
                 32,
                 &[200, 200, 100],
                 500,
     )
}

pub fn rustception_optimized_v2() -> Config<'static> {
     Config::new(0.01,
                 32,
                 0, // no limit
                 ml_200_tok_vocab_v3(),
                 32,
                 32,
                 &[250, 200, 64],
                 500,
     )
}

pub fn rustception_optimized_v2_train_options(lr: f32, batch_size: usize, epochs: usize) -> Config<'static> {
     Config::new(lr,
                 batch_size,
                 0, // no limit
                 ml_200_tok_vocab_v3(),
                 32,
                 32,
                 &[250, 200, 64],
                 epochs,
     )
}

pub fn rustception_optimized_v2_large_train_options(lr: f32, batch_size: usize, epochs: usize) -> Config<'static> {
     Config::new(lr,
                 batch_size,
                 0, // no limit
                 ml_200_tok_vocab_v3(),
                 32,
                 42,
                 &[250, 200, 100, 64],
                 epochs,
     )
}

pub fn rustception_v3_to(lr: f32, batch_size: usize, epochs: usize) -> Config<'static> { // to is short for train options
     Config::new(lr,
                 batch_size,
                 0, // no limit
                 ml_v4(),
                 32,
                 42,
                 &[250, 200, 100, 64],
                 epochs,
     )
}

pub fn rustception_v4_mini_to(lr: f32, batch_size: usize, epochs: usize) -> Config<'static> { // to is short for train options
     Config::new(lr,
                 batch_size,
                 0, // no limit
                 ml_v4(),
                 16,
                 32,
                 &[200, 100],
                 epochs,
     )
}

pub fn poke_v1_mini_to(lr: f32, batch_size: usize, epochs: usize) -> Config<'static> { // to is short for train options
    Config::new(lr,
                batch_size,
                0, // no limit
                poke_v1(),
                16,
                40,
                &[200, 100],
                epochs,
    )
}

pub fn tale_v1_mini_to(lr: f32, batch_size: usize, epochs: usize) -> Config<'static> { // to is short for train options
    Config::new(lr,
                batch_size,
                0, // no limit
                tale_v1(),
                16,
                40,
                &[200, 100],
                epochs,
    )
}

pub fn tale_v1_mini_mini_to(lr: f32, batch_size: usize, epochs: usize) -> Config<'static> { // to is short for train options
    Config::new(lr,
                batch_size,
                0, // no limit
                tale_v1(),
                16,
                32,
                &[100],
                epochs,
    )
}

pub fn tale_v1_scout_to(lr: f32, batch_size: usize, epochs: usize) -> Config<'static> { // to is short for train options
    Config::new(lr,
                batch_size,
                0, // no limit
                tale_v1(),
                32,
                30,
                &[90],
                epochs,
    )
}

pub fn poke_v2_mini_to(lr: f32, batch_size: usize, epochs: usize) -> Config<'static> { // to is short for train options
    Config::new(lr,
                batch_size,
                0, // no limit
                poke_v1(),
                16,
                32,
                &[100],
                epochs,
    )
}

pub fn recipe_v1_to(lr: f32, batch_size: usize, epochs: usize) -> Config<'static> { // to is short for train options
    Config::new(lr,
                batch_size,
                0, // no limit
                recipe_v1(),
                16,
                40,
                &[200, 100],
                epochs,
    )
}

pub fn poke_v3_behemoth_to(lr: f32, batch_size: usize, epochs: usize) -> Config<'static> { // to is short for train options
    Config::new(lr,
                batch_size,
                0, // no limit
                poke_v2(),
                64,
                40,
                &[400, 150],
                epochs,
    )
}

pub fn poke_v3_to(lr: f32, batch_size: usize, epochs: usize) -> Config<'static> { // to is short for train options
    Config::new(lr,
                batch_size,
                0, // no limit
                poke_v2(), // 381 tokens, 300 are multi-char
                64,
                30,
                &[100],
                epochs,
    )
}

pub fn poke_v4_32_context_to(lr: f32, batch_size: usize, epochs: usize) -> Config<'static> { // to is short for train options
    Config::new(lr,
                batch_size,
                0, // no limit
                poke_v2(), // 381 tokens, 300 are multi-char
                32,
                30,
                &[100, 100, 100],
                epochs,
    )
}

pub fn oasst1_v1_to(lr: f32, batch_size: usize, epochs: usize) -> Config<'static> { // to is short for train options
    Config::new(lr,
                batch_size,
                0, // no limit
                vocabs::oasst1(),
                128,
                64,
                &[1024, 512, 512, 64],
                epochs,
    )
}

pub fn fineweb_v1_to(lr: f32, batch_size: usize, epochs: usize) -> Config<'static> { // to is short for train options
    Config::new(lr,
                batch_size,
                0, // no limit
                vocabs::fineweb(),
                128,
                64,
                &[512, 512, 64],
                epochs,
    )
}

pub fn fineweb_v2_to(lr: f32, batch_size: usize, epochs: usize) -> Config<'static> { // to is short for train options
    Config::new(lr,
                batch_size,
                0, // no limit
                vocabs::fineweb_v2(),
                128,
                64,
                &[512, 512, 64],
                epochs,
    )
}
