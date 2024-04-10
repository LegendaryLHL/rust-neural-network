use rand::{thread_rng, Rng};
use std::f64;

#[derive(Clone)]
struct Neuron {
    delta: f64,
    weighted_input: f64,
    weights: Vec<f64>,
    num_weights: i32,
    bias: f64,
    output: f64,
}

#[derive(Clone)]
struct Layer {
    size: i32,
    neurons: Vec<Neuron>,
}

impl Layer {
    fn compute(&mut self, previous_layer: &Layer, activation_function: fn(f64, bool) -> f64) {
        for neuron in &mut self.neurons {
            neuron.output = 0.0;
            neuron.weighted_input = 0.0;
            for i in 0..previous_layer.size {
                neuron.weighted_input +=
                    previous_layer.neurons[i as usize].output * neuron.weights[i as usize];
            }
            neuron.weighted_input += neuron.bias;
            neuron.output = activation_function(neuron.weighted_input, false);
        }
    }
}

struct NeuralNetwork {
    input_layer: Layer,
    hidden_layers: Vec<Layer>,
    output_layer: Layer,
    num_hidden_layer: i32,
    activation_function: fn(f64, bool) -> f64,
}

impl NeuralNetwork {
    fn new(
        hidden_layer_sizes: Vec<i32>,
        output_layer_size: i32,
        input_layer_size: i32,
        activation_function: fn(f64, bool) -> f64,
    ) -> Self {
        let mut input_neurons = Vec::with_capacity(input_layer_size as usize);
        for _ in 0..input_layer_size {
            input_neurons.push(Neuron {
                delta: 0.0,
                weighted_input: 0.0,
                weights: Vec::new(),
                num_weights: 0,
                bias: 0.0,
                output: 0.0,
            });
        }

        let mut rng = thread_rng();
        let mut hidden_layers = Vec::with_capacity(hidden_layer_sizes.len());
        for &size in hidden_layer_sizes.iter() {
            let mut hidden_layer_neurons = Vec::with_capacity(size as usize);
            let weights: Vec<f64>;
            if size == 0 {
                weights = vec![rng.gen_range(-1.0..1.0); input_layer_size as usize];
            } else {
                let prev_layer_size = hidden_layer_sizes[size as usize - 1] as usize;
                weights = vec![rng.gen_range(-1.0..1.0); prev_layer_size];
            }
            for _ in 0..size {
                hidden_layer_neurons.push(Neuron {
                    delta: 0.0,
                    weighted_input: 0.0,
                    weights: weights.clone(),
                    num_weights: weights.len() as i32,
                    bias: 0.0,
                    output: 0.0,
                });
            }
            hidden_layers.push(Layer {
                size,
                neurons: hidden_layer_neurons,
            });
        }

        let mut output_neurons = Vec::with_capacity(output_layer_size as usize);
        let weights: Vec<f64>;
        if hidden_layer_sizes.len() == 0 {
            weights = vec![rng.gen_range(-1.0..1.0); input_layer_size as usize];
        } else {
            weights = vec![
                rng.gen_range(-1.0..1.0);
                hidden_layer_sizes[hidden_layer_sizes.len() as usize - 1] as usize
            ];
        }
        for _ in 0..output_layer_size {
            output_neurons.push(Neuron {
                delta: 0.0,
                weighted_input: 0.0,
                weights: weights.clone(),
                num_weights: 0,
                bias: 0.0,
                output: 0.0,
            });
        }

        NeuralNetwork {
            input_layer: Layer {
                size: input_layer_size,
                neurons: input_neurons,
            },
            hidden_layers,
            output_layer: Layer {
                size: output_layer_size,
                neurons: output_neurons,
            },
            num_hidden_layer: hidden_layer_sizes.len() as i32,
            activation_function,
        }
    }

    fn compute(&mut self, inputs: Vec<f64>) {
        for i in 0..self.input_layer.size {
            self.input_layer.neurons[i as usize].output = inputs[i as usize];
        }

        if self.num_hidden_layer == 0 {
            self.output_layer
                .compute(&self.input_layer, self.activation_function);
        } else {
            for i in 0..self.num_hidden_layer {
                if i == 0 {
                    self.hidden_layers[i as usize]
                        .compute(&self.input_layer, self.activation_function);
                } else {
                    let previous_layer = self.hidden_layers[i as usize - 1].clone();
                    self.hidden_layers[i as usize]
                        .compute(&previous_layer, self.activation_function);
                }
            }
        }
    }
}

const IMAGE_SIZE: usize = 28;
struct Data {
    inputs: [f64; IMAGE_SIZE * IMAGE_SIZE],
    expected: f64,
}

fn max(a: f64, b: f64) -> f64 {
    if a > b {
        return a;
    }
    return b;
}

fn sigmoid(val: f64, is_deravative: bool) -> f64 {
    let result = 1.0 / (1.0 + f64::exp(-val));
    if is_deravative {
        return result * (1.0 - result);
    }
    return result;
}

fn relu(val: f64, is_deravative: bool) -> f64 {
    if is_deravative {
        if val > 0.0 {
            return 1.0;
        }
        return 0.0;
    }
    return max(0.0, val);
}

fn output_expected(neuron_index: i32, data: &Data) -> f64 {
    if data.expected == neuron_index as f64 {
        return 1.0;
    }
    return 0.0;
}

fn main() {
    println!("Hello, world!");
}
