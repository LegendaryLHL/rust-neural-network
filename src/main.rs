use rand::{thread_rng, Rng};
use std::{f64, thread::JoinHandle};

#[derive(Clone)]
struct Neuron {
    delta: f64,
    weighted_input: f64,
    weights: Vec<f64>,
    bias: f64,
    output: f64,
}

struct NeuralNetwork {
    input_layer: Vec<Neuron>,
    hidden_layers: Vec<Vec<Neuron>>,
    output_layer: Vec<Neuron>,
    activation_function: fn(f64, bool) -> f64,
}
impl NeuralNetwork {
    fn new(
        hidden_layer_sizes: Vec<i32>,
        output_layer_size: i32,
        input_layer_size: i32,
        activation_function: fn(f64, bool) -> f64,
    ) -> Self {
        let mut input_layer = Vec::with_capacity(input_layer_size as usize);
        for _ in 0..input_layer_size {
            input_layer.push(Neuron {
                delta: 0.0,
                weighted_input: 0.0,
                weights: Vec::new(),
                bias: 0.0,
                output: 0.0,
            });
        }

        let mut rng = thread_rng();
        let mut hidden_layers = Vec::with_capacity(hidden_layer_sizes.len());
        for &size in hidden_layer_sizes.iter() {
            let mut hidden_layer = Vec::with_capacity(size as usize);
            let weights: Vec<f64>;
            if size == 0 {
                weights = vec![rng.gen_range(-1.0..1.0); input_layer_size as usize];
            } else {
                let prev_layer_size = hidden_layer_sizes[size as usize - 1] as usize;
                weights = vec![rng.gen_range(-1.0..1.0); prev_layer_size];
            }
            for _ in 0..size {
                hidden_layer.push(Neuron {
                    delta: 0.0,
                    weighted_input: 0.0,
                    weights: weights.clone(),
                    bias: 0.0,
                    output: 0.0,
                });
            }
            hidden_layers.push(hidden_layer);
        }

        let mut output_layer = Vec::with_capacity(output_layer_size as usize);
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
            output_layer.push(Neuron {
                delta: 0.0,
                weighted_input: 0.0,
                weights: weights.clone(),
                bias: 0.0,
                output: 0.0,
            });
        }

        NeuralNetwork {
            input_layer,
            hidden_layers,
            output_layer,
            activation_function,
        }
    }

    fn softmax(&mut self) -> Vec<f64> {
        let mut sum = 0.0;
        for i in 0..self.output_layer.len() {
            sum += f64::exp(self.output_layer[i].output);
        }
        let mut softmax_values = Vec::with_capacity(self.output_layer.len());
        for i in 0..self.output_layer.len() {
            let softmax_val = f64::exp(self.output_layer[i].output) / sum;
            softmax_values.push(softmax_val);
        }

        softmax_values
    }

    fn print_percentages(&mut self) {
        let softmax_values = self.softmax();
        for i in 0..softmax_values.len() {
            print!("{} Percentage: {:.2}%. ", i, softmax_values[i] * 100.0);
        }
        println!();
    }
    fn compute(&mut self, inputs: Vec<f64>) {
        // Add input
        for i in 0..self.input_layer.len() {
            self.input_layer[i].output = inputs[i];
        }

        // Feed foward
        if self.hidden_layers.len() == 0 {
            compute_layer(
                &mut self.output_layer,
                &self.input_layer,
                self.activation_function,
            );
        } else {
            for i in 0..self.hidden_layers.len() {
                if i == 0 {
                    compute_layer(
                        &mut self.hidden_layers[i],
                        &self.input_layer,
                        self.activation_function,
                    );
                } else {
                    let previous_layer = self.hidden_layers[i - 1].clone();
                    compute_layer(
                        &mut self.hidden_layers[i],
                        &previous_layer,
                        self.activation_function,
                    );
                }
            }
        }
    }
    fn cost(&mut self, datas: &Vec<Data>) -> f64 {
        let mut cost = 0.0;
        for data in datas {
            self.compute(data.inputs.clone());
            for i in 0..self.output_layer.len() {
                let neuron_output = self.output_layer[i].output;
                cost += (neuron_output - output_expected(i as i32, data))
                    * (neuron_output - output_expected(i as i32, data));
            }
        }
        return cost / datas.len() as f64;
    }

    fn learn(&mut self, learn_rate: f64, training_data: &Vec<Data>) {
        for data in training_data {
            self.compute(data.inputs.clone());
            for i in 0..self.output_layer.len() {
                let neuron = &mut self.output_layer[i];
                let expected = output_expected(i as i32, data);

                neuron.delta = 2.0
                    * (neuron.output - expected)
                    * (self.activation_function)(neuron.weighted_input, true);

                for j in 0..neuron.weights.len() {
                    let input = self.hidden_layers[self.hidden_layers.len() - 1][j].output;
                    neuron.weights[j] -= neuron.delta * input * learn_rate;
                }

                neuron.bias -= neuron.delta * learn_rate;
            }

            for i in self.hidden_layers.len() - 1..=0 {
                let previous_layer;
                let next_layer;
                if i == 0 {
                    previous_layer = &self.input_layer;
                } else {
                    previous_layer = &self.hidden_layers[i - 1];
                }

                if i == self.hidden_layers.len() - 1 {
                    next_layer = &self.output_layer;
                }

                for j in 0..self.hidden_layers[i].len() {}
            }
        }
    }
}

struct Data {
    inputs: Vec<f64>,
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
fn compute_layer(
    layer: &mut Vec<Neuron>,
    previous_layer: &Vec<Neuron>,
    activation_function: fn(f64, bool) -> f64,
) {
    for neuron in layer {
        neuron.output = 0.0;
        neuron.weighted_input = 0.0;
        for i in 0..previous_layer.len() {
            neuron.weighted_input += previous_layer[i].output * neuron.weights[i];
        }
        neuron.weighted_input += neuron.bias;
        neuron.output = activation_function(neuron.weighted_input, false);
    }
}
fn main() {
    println!("Hello, world!");
}
