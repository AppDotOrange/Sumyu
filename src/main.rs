use grad::fnn_lm::{SavedLM, LM};
use grad::helper;
use std::{fs, io};
use std::io::Write;
use grad::helper::{poke_v2, recipe_v3};
use grad::neuron::{Activation, LayerSpec};
use grad::trainer::CheckpointFrequency;

fn main() {
    print!("\x1B[2J\x1B[3J\x1B[1;1H");
    let _ = io::stdout().flush();
    // to save ram:
    let _rust = "";
    let text = "";
    let poke = "";
    let recipe = "";
    let recipe_full = "";
    let oasst1 = "";
    let fineweb = "";
    let fineweb2 = "";
    let fineweb3 = "";

    //let _rust = fs::read_to_string("Datasets/rust.txt").expect("Can't read rust.txt!");
    //let text = fs::read_to_string("Datasets/Grimm's Fairy Tales").expect("Can't read Grimm's Fairy Tales!").replace("\r\n", "\n");
    //let poke = fs::read_to_string("Datasets/pokedex.txt").expect("Can't read pokedex.txt!").replace("\r\n", "\n");
    //let recipe = fs::read_to_string("Datasets/150recipes.txt").expect("Can't read 150recipes.txt!").replace("\r\n", "\n");
    //let recipe_full = fs::read_to_string("Datasets/recipes.txt").expect("Can't read recipes.txt!").replace("\r\n", "\n");
    //let oasst1 = fs::read_to_string("Datasets/oasst1.txt").expect("Can't read oasst1.txt!").replace("\r\n", "\n");
    //let fineweb = fs::read_to_string("Datasets/fineweb.txt").expect("Can't read fineweb.txt!").replace("\r\n", "\n");
    //let fineweb2 = fs::read_to_string("Datasets/fineweb_v2.txt").expect("Can't read fineweb_v2.txt!");
    let fineweb3 = fs::read_to_string("Datasets/fineweb_v3.txt").expect("Can't read fineweb_v3.txt!");
    let test = 16;
    if test == -3 {
        #[cfg(target_arch = "x86_64")]
        println!(
            "AVX2: {}, FMA: {}",
            is_x86_feature_detected!("avx2"),
            is_x86_feature_detected!("fma"),
        );
    } else if test == -2 {
        println!("{}", poke_v2().len())
    } else if test == -1 {
        //------------------------------------------------------------------------------------------
        //     CONFIG
        //------------------------------------------------------------------------------------------
        
        let lr = 0.01;
        let batch_size = 32;
        let max_batches_per_epoch = 10; // 0 means no limit
        let vocab = helper::ml_200_tok_vocab_v3();
        let context_len = 8;
        let emb_dim = 42;
        let hidden_dim: &[usize] = &[200];
        let epochs = 500;

        //------------------------------------------------------------------------------------------
        //     DON'T TOUCH
        //------------------------------------------------------------------------------------------
        let mut lm = LM::new(context_len, vocab, hidden_dim, emb_dim);
        lm.params();
        lm.load_corpus(&poke);
        lm.train_options(lr, epochs, batch_size, max_batches_per_epoch);
        //lm.train(None);
    } else if test == 0 {
        //------------------------------------------------------------------------------------------
        //     CONFIG
        //------------------------------------------------------------------------------------------

        let lr = 0.01;
        let batch_size = 32;
        let max_batches_per_epoch = 0; // 0 means no limit
        let vocab = helper::ml_200_tok_vocab_v3();
        let context_len = 16;
        let emb_dim = 42;
        let hidden_dim: &[usize] = &[400, 200];
        let epochs = 500;

        let load = true;

        //------------------------------------------------------------------------------------------
        //     DON'T TOUCH
        //------------------------------------------------------------------------------------------
        if load {
            let bytes = fs::read("model.sumyu").unwrap();
            let (model, _): (SavedLM, usize) =
                bincode::serde::decode_from_slice(
                    &bytes,
                    bincode::config::standard(),
                ).unwrap();
            let mut lm = LM::from_saved(model);
            println!("Loaded!");
            lm.params();
            lm.load_corpus(&text);
            lm.train_options(lr, epochs, batch_size, max_batches_per_epoch);
            //lm.train(None);
        } else {
            let mut lm = LM::new(context_len, vocab, hidden_dim, emb_dim);
            lm.params();
            lm.load_corpus(&text);
            lm.train_options(lr, epochs, batch_size, max_batches_per_epoch);
            //lm.train(None);
        let saved = lm.to_saved("");
        let bytes = bincode::serde::encode_to_vec(
            &saved,
            bincode::config::standard(),
        ).unwrap();
        fs::write("model.sumyu", bytes).unwrap();
        }
    } else if test == 1 {
        
    } else if test == 2 {
        let bytes = fs::read("ChatterP1.sumyu").unwrap();
        let (model, _): (SavedLM, usize) =
            bincode::serde::decode_from_slice(
                &bytes,
                bincode::config::standard(),
            ).unwrap();
        let lm = LM::from_saved(model);
        lm.params();
        //lm.generate_gpt("<USER>Can you write a story about two birds?<EOT>\n".to_string(), 1000, 0.7);
        //lm.generate_one_distribution("".to_string(), 25);
        //println!("\"{}\"", lm.generate("pub fn ".to_string(), 100, 0.7))
        //lm.embeds().find_clusters(0.50, helper::recipe_v1());
    } else if test == 3 {
        //------------------------------------------------------------------------------------------
        //     CONFIG
        //------------------------------------------------------------------------------------------

        //let config = helper::rustception_v4_mini_to(0.01, 32, 500);

        //------------------------------------------------------------------------------------------
        //     DON'T TOUCH
        //------------------------------------------------------------------------------------------
        //let mut lm = LM::from_config(config);
        //*
        let (description, mut lm) = LM::load("Production/Rustception_P1_mini.sumyu");
        lm.train_options(0.01, 100_000, 32, 0);
        //*/
        lm.params();
        lm.load_corpus(&text);
        //lm.train(None);
        lm.save(
            "Rustception_optimized.sumyu",
            &description,
        );
    } else if test == 5 {
        //------------------------------------------------------------------------------------------
        //     CONFIG
        //------------------------------------------------------------------------------------------

        let config = helper::poke_v4_32_context_to(0.1, 32, 100_000);
        let description = "A Sumyu model trained on a filtered Pokedex.";

        //------------------------------------------------------------------------------------------
        //     DON'T TOUCH
        //------------------------------------------------------------------------------------------
        let mut lm = LM::from_config(config);
        /*
        let (description, mut lm) = LM::load("Production/PokeP1_64_minutes.sumyu");
        lm.train_options(0.05, 100_000, 32, 0);
        */

        lm.params();
        lm.load_corpus(&poke);
        //lm.train(None);
        lm.save(
            "PokeP4_32c.sumyu",
            &description,
        );
    } else if test == 6 {
        //------------------------------------------------------------------------------------------
        //     CONFIG
        //------------------------------------------------------------------------------------------

        //let config = helper::tale_v1_scout_to(0.001, 32, 100_000);
        //let description = "A Sumyu model trained on Grimm's Fairy Tales, obtained from Project Gutenberg.";

        //------------------------------------------------------------------------------------------
        //     DON'T TOUCH
        //------------------------------------------------------------------------------------------
        //let mut lm = LM::from_config(config);
        //*
        let (description, mut lm) = LM::load("Tests/Tale_V1_scout_10.sumyu");
        lm.train_options(0.001, 100_000, 32, 0);
        //*/
        lm.params();
        lm.load_corpus(&text);
        //lm.train(None);
        lm.save(
            "Tale_V1.sumyu",
            &description,
        );
    } else if test == 7 {
        //------------------------------------------------------------------------------------------
        //     CONFIG
        //------------------------------------------------------------------------------------------

        let config = helper::recipe_v3_to(0.1, 128, 100_000);
        let description = "A Sumyu model trained on recipes.";

        //------------------------------------------------------------------------------------------
        //     DON'T TOUCH
        //------------------------------------------------------------------------------------------
        unsafe extern "C" {
            fn openblas_set_num_threads(num_threads: i32);
            fn openblas_get_num_threads() -> i32;
        }
        unsafe {
            openblas_set_num_threads(3);
            println!("OpenBLAS threads: {}", openblas_get_num_threads());
        }
        let mut lm = LM::from_config(config);
        lm.params();
        lm.load_corpus(&recipe_full);
        lm.train(None, None, None, CheckpointFrequency::Disabled, None, None);
        lm.save(
            "RecipeP4.sumyu",
            &description,
        );
    } else if test == 8 {
        // let mut terminal = sumyu::Sumyu::new();
        // terminal.start();
    } else if test == 9 {
        //------------------------------------------------------------------------------------------
        //     CONFIG
        //------------------------------------------------------------------------------------------

        //let config = helper::oasst1_v1_to(0.1, 128, 10);
        //let description = "A huge Sumyu model trained on dialogue (oasst1 dataset).";

        //------------------------------------------------------------------------------------------
        //     DON'T TOUCH
        //------------------------------------------------------------------------------------------
        //let mut lm = LM::from_config(config);
        unsafe extern "C" {
            fn openblas_set_num_threads(num_threads: i32);
            fn openblas_get_num_threads() -> i32;
        }
        unsafe {
            openblas_set_num_threads(4);
            println!("OpenBLAS threads: {}", openblas_get_num_threads());
        }
        //*
        let (description, mut lm) = LM::load("Production/ChatterP1-preview.sumyu");
        lm.train_options(0.01, 10, 128, 0);
        //*/
        lm.params();
        lm.load_corpus(&oasst1);
        lm.train(
            Some(10),
            Some("chatterP1_checks/ChatterV1_batch_47443_epoch_2.check"),
            Some("chatterP1_checks/ChatterV1"),
            CheckpointFrequency::EveryBatch(1000),
            Some(0.01),
            None,
        );
        lm.save(
            "ChatterP2.sumyu",
            &description,
        );
    } else if test == 10 {
        unsafe extern "C" {
            fn openblas_set_num_threads(num_threads: i32);
            fn openblas_get_num_threads() -> i32;
        }
        unsafe {
            openblas_set_num_threads(2);
            println!("OpenBLAS threads: {}", openblas_get_num_threads());
        }
    } else if test == 11 {
        //------------------------------------------------------------------------------------------
        //     CONFIG
        //------------------------------------------------------------------------------------------

        let config = helper::fineweb_v2_to(0.01, 256, 1);
        let description = "A large Sumyu model pre-trained on the FineWeb dataset family.";

        //------------------------------------------------------------------------------------------
        //     DON'T TOUCH
        //------------------------------------------------------------------------------------------
        let mut lm = LM::from_config(config);
        unsafe extern "C" {
            fn openblas_set_num_threads(num_threads: i32);
            fn openblas_get_num_threads() -> i32;
        }
        unsafe {
            openblas_set_num_threads(4);
            println!("OpenBLAS threads: {}", openblas_get_num_threads());
        }
        /*
        let (description, mut lm) = LM::load("Production/ChatterP1-preview.sumyu");
        lm.train_options(0.01, 1, 256, 0);
        */
        lm.params();
        lm.load_corpus(&fineweb);
        lm.train(
            Some(10),
            Some("pretraining/pretrainV2_batch_88714_epoch_1.check"),
            Some("pretraining/pretrainV2"),
            CheckpointFrequency::EveryBatch(1000),
            Some(0.08),
            None,
        );
        lm.save(
            "pretrainV2.sumyu",
            description,
        );
    } else if test == 12 {
        unsafe extern "C" {
            fn openblas_set_num_threads(num_threads: i32);
            fn openblas_get_num_threads() -> i32;
        }
        unsafe {
            openblas_set_num_threads(4);
            println!("OpenBLAS threads: {}", openblas_get_num_threads());
        }
        //*
        let description = "Conv1D recipe experiment.";

        let mut lm = LM::from_layers(
            32,
            recipe_v3(),
            &[
                // Initial feature extraction.
                LayerSpec::Conv1D {
                    in_channels: 30,
                    out_channels: 96,
                    kernel_size: 3,
                    stride: 1,
                    padding: 1,
                    causal: false,
                    activation: Activation::LeakyReLU { slope: 0.01 },
                },

                LayerSpec::ChannelScale {
                    channels: 96,
                },

                // Residual feature-processing block.
                LayerSpec::Residual {
                    layers: vec![
                        LayerSpec::DepthwiseConv1D {
                            in_channels: 96,
                            kernel_size: 3,
                            stride: 1,
                            padding: 1,
                            causal: false,
                            activation: Activation::LeakyReLU { slope: 0.01 },
                        },

                        LayerSpec::ChannelScale {
                            channels: 96,
                        },

                        LayerSpec::Conv1D {
                            in_channels: 96,
                            out_channels: 96,
                            kernel_size: 1,
                            stride: 1,
                            padding: 0,
                            causal: false,
                            activation: Activation::LeakyReLU { slope: 0.01 },
                        },

                        LayerSpec::ChannelScale {
                            channels: 96,
                        },
                    ],
                },

                // Downsample and reduce width.
                LayerSpec::Conv1D {
                    in_channels: 96,
                    out_channels: 64,
                    kernel_size: 3,
                    stride: 2,
                    padding: 1,
                    causal: false,
                    activation: Activation::LeakyReLU { slope: 0.01 },
                },

                LayerSpec::ChannelScale {
                    channels: 64,
                },

                // Second residual feature-processing block.
                LayerSpec::Residual {
                    layers: vec![
                        LayerSpec::DepthwiseConv1D {
                            in_channels: 64,
                            kernel_size: 3,
                            stride: 1,
                            padding: 1,
                            causal: false,
                            activation: Activation::LeakyReLU { slope: 0.01 },
                        },

                        LayerSpec::ChannelScale {
                            channels: 64,
                        },

                        LayerSpec::Conv1D {
                            in_channels: 64,
                            out_channels: 64,
                            kernel_size: 1,
                            stride: 1,
                            padding: 0,
                            causal: false,
                            activation: Activation::LeakyReLU { slope: 0.01 },
                        },

                        LayerSpec::ChannelScale {
                            channels: 64,
                        },
                    ],
                },

                // Final downsampling.
                LayerSpec::Conv1D {
                    in_channels: 64,
                    out_channels: 32,
                    kernel_size: 3,
                    stride: 2,
                    padding: 1,
                    causal: false,
                    activation: Activation::LeakyReLU { slope: 0.01 },
                },

                LayerSpec::ChannelScale {
                    channels: 32,
                },

                // Bottleneck.
                LayerSpec::Dense {
                    output_size: 30,
                    activation: Activation::LeakyReLU { slope: 0.01 },
                },

                // Vocabulary projection.
                LayerSpec::Dense {
                    output_size: recipe_v3().len(),
                    activation: Activation::None,
                },
            ],
            30,
        );
        //*/
        //let (description, mut lm) = LM::load("MiniRecipe.sumyu");

        lm.train_options(0.5, 100_000, 128, 0);
        lm.params();
        lm.load_corpus(&recipe);
        lm.train(
            None,
            None,
            None,
            CheckpointFrequency::Disabled,
            None,
            None,
        );
        lm.save("MiniRecipe.sumyu", &description);
    } else if test == 13 {
        //------------------------------------------------------------------------------------------
        //     CONFIG
        //------------------------------------------------------------------------------------------

        let config = helper::fineweb_hybrid_v4_to(0.4, 256, 1);
        let description = "A large Sumyu Hybrid model pre-trained on the FineWeb dataset family.";

        //------------------------------------------------------------------------------------------
        //     DON'T TOUCH
        //------------------------------------------------------------------------------------------
        let mut lm = LM::from_hybrid_config(config);
        unsafe extern "C" {
            fn openblas_set_num_threads(num_threads: i32);
            fn openblas_get_num_threads() -> i32;
        }
        unsafe {
            openblas_set_num_threads(4);
            println!("OpenBLAS threads: {}", openblas_get_num_threads());
        }
        /*
        let (description, mut lm) = LM::load("pretrainV2.sumyu");
        lm.train_options(0.55, 1, 256, 0);
        */
        lm.params();
        lm.load_corpus(&fineweb3);
        lm.train(
            Some(10),
            Some("pretraining/pretrainConV4_batch_47243_epoch_1.check"),
            Some("pretraining/pretrainConV4"),
            CheckpointFrequency::EveryBatch(1000),
            Some(0.3),
            None,
        );
        lm.save(
            "pretrainV4.sumyu",
            &description,
        );
    } else if test == 14 {
        //------------------------------------------------------------------------------------------
        //     CONFIG
        //------------------------------------------------------------------------------------------

        let config = helper::fineweb_hybrid_v5_to(0.3, 256, 1);
        let description = "A large Sumyu Hybrid model pre-trained on the FineWeb dataset family.";

        //------------------------------------------------------------------------------------------
        //     DON'T TOUCH
        //------------------------------------------------------------------------------------------
        let mut lm = LM::from_hybrid_config(config);
        unsafe extern "C" {
            fn openblas_set_num_threads(num_threads: i32);
            fn openblas_get_num_threads() -> i32;
        }
        unsafe {
            openblas_set_num_threads(4);
            println!("OpenBLAS threads: {}", openblas_get_num_threads());
        }
        /*
        let (description, mut lm) = LM::load("pretrainV2.sumyu");
        lm.train_options(0.55, 1, 256, 0);
        */
        lm.params();
        lm.load_corpus(&fineweb3);
        lm.train(
            Some(10),
            Some("pretraining/pretrainConV5_batch_46329_epoch_1.check"),
            Some("pretraining/pretrainConV5"),
            CheckpointFrequency::EveryBatch(1000),
            Some(0.2),
            None,
        );
        lm.save(
            "pretrainV5.sumyu",
            &description,
        );
    } else if test == 15 {
        //------------------------------------------------------------------------------------------
        //     CONFIG
        //------------------------------------------------------------------------------------------

        let config = helper::fineweb_hybrid_v6_to(0.3, 256, 1);
        let description = "A large Sumyu Hybrid model pre-trained on the FineWeb dataset family.";

        //------------------------------------------------------------------------------------------
        //     DON'T TOUCH
        //------------------------------------------------------------------------------------------
        let mut lm = LM::from_hybrid_config(config);
        unsafe extern "C" {
            fn openblas_set_num_threads(num_threads: i32);
            fn openblas_get_num_threads() -> i32;
        }
        unsafe {
            openblas_set_num_threads(4);
            println!("OpenBLAS threads: {}", openblas_get_num_threads());
        }
        /*
        let (description, mut lm) = LM::load("pretrainV2.sumyu");
        lm.train_options(0.55, 1, 256, 0);
        */
        lm.params();
        lm.load_corpus(&fineweb3);
        lm.train(
            Some(10),
            Some("pretraining_v6/pretrainConV6_batch_1130_epoch_1.check"),
            Some("pretraining_v6/pretrainConV6"),
            CheckpointFrequency::EveryBatch(1000),
            Some(0.3),
            None,
        );
        lm.save(
            "pretrainV6.sumyu",
            &description,
        );
    } else if test == 16 {
        //------------------------------------------------------------------------------------------
        //     CONFIG
        //------------------------------------------------------------------------------------------

        let config = helper::fineweb_hybrid_v8_to(0.4, 256, 1);
        let description = "A large Sumyu Hybrid model pre-trained on the FineWeb dataset family.";

        //------------------------------------------------------------------------------------------
        //     DON'T TOUCH
        //------------------------------------------------------------------------------------------
        let mut lm = LM::from_hybrid_config(config);
        unsafe extern "C" {
            fn openblas_set_num_threads(num_threads: i32);
            fn openblas_get_num_threads() -> i32;
        }
        unsafe {
            openblas_set_num_threads(4);
            println!("OpenBLAS threads: {}", openblas_get_num_threads());
        }
        /*
        let (description, mut lm) = LM::load("pretrainV2.sumyu");
        lm.train_options(0.55, 1, 256, 0);
        */
        lm.params();
        lm.load_corpus(&fineweb3);
        lm.train(
            Some(10),
            None,
            Some("pretraining_v8/pretrainConV8"),
            CheckpointFrequency::EveryBatch(500),
            None,
            None,
        );
        lm.save(
            "pretrainV8.sumyu",
            &description,
        );
    }
}
