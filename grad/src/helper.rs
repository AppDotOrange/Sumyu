use std::collections::HashMap;
use crate::fnn_lm::{tokenize};

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

pub fn tale_v1() -> Vec<String> { // 250 tokens
    ["<UNKNOWN>", "", "T", "H", "E", " ", "F", "R", "O", "G", "-", "K", "I", "N", ";", ",", "Y", "\n", "n", "o", "l", "d", "t", "i", "m", "e", "s", "w", "h", "g", "a", "v", "r", "u", "b", "f", "y", "c", ".", "C", "’", "k", "W", "p", "A", "D", "“", "?", "”", "S", "!", "B", "q", "j", "_", "x", ":", "M", "L", "[", "]", "V", "U", "P", "Z", "(", "z", ")", "‘", "J", "Q", "<", ">", "X", "e ", "th", "d ", " th", " a", "er", " the ", "t ", "nd ", "in", " and ", ", ", "s ", "and ", "ou", "en", "he ", "to", "the ", "ed ", "ing", "\n\n", " h", "on", "an", " s", "or", "ll", "ea", " w", "as ", ". ", "at ", "y ", "you", "  ", "hi", "e, ", "ai", "no", "d, ", "ther", "hen", ".\n\n", "”\n\n", "for", "re", "of", "ha", "it", "om", "her", "to ", "st", "the", "ow", "gh", "ll ", "ut ", "was ", ".”\n\n", "ld ", "se", "ing ", "le ", "ver", "ve ", "ch", "ith", " the", "be", "not ", "is ", "ca", "The ", " to ", "she ", " m", "le", "oo", "wa", "had ", "s, ", "    ", " you", "ill ", "with", "aid ", "me ", "ter", "that ", "ck", "ri", "d\n", "sh", "la", "out ", "ti", " was ", "ould ", "ir", " in", "other", " ha", " she ", "li", "him", "a ", "un", "tle ", "ear", "but ", "one ", "t, ", "ght ", "ever", " her", "ly ", "ent ", "ar", "go", "ain", "aid, ", "Then", " he ", ",\n", "when", " it", " said ", "his ", "e\n", "ttle ", "little ", "came ", "es ", "hat ", "ho", " f", " d", "y, ", "ed, ", "rea", "they ", "ng", "said, ", " his ", "all ", "wer", " of", "sel", " be", "ra", " b", "ld", "up", "ke ", "went ", "ust", "But ", " that ", "And ", ",”", " will ", "When", "so", "ing, ", "self", "ce ", " had ", "ed", "own", "gain", "hich", "too", "if", "who", "rou", " they ", "thing ", "est", "com", "ro", "’s ", "here ", " with", "id ", "have ", " p", "sa", "      ", "now", "ven", "and\n", "es", "ked ", "into ", "were ", "ep", "ta", "t\n", "et ", " for", "el", "ell", "fu", "man", "e. ", "ong", "im", "der", "“I", " on", "long", "si", " a ", "ted ", "s\n", "da", " him", "ill", " as ", "could ", "took", "ur", "ne", "rom", "the\n", "King ", "de ", ",\n      ", "them", "pp", " an", "mor", "over", "which", "al", "oman", "ish", "King", "She ", "will ", " them", "did ", " into ", "oor", " have ", " the\n", "ghter", "ughter"].iter().map(|x| { String::from(*x) }).collect()
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
              let len = pair.chars().count() as f32;
              (*count as f32 * len.sqrt()) as usize
         })
}

pub fn make_vocab(text: &str, token_num: usize) -> Vec<String> {
     let mut vocab: Vec<String> = vec!["<UNKNOWN>".to_string(), "".to_string()];
     let mut chars = Vec::new();
     for x in text.chars() {
          let y = x.to_string();
          if !chars.contains(&y) {
               chars.push(y)
          }
     }
     let len = chars.len()+2;
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
