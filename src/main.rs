extern crate rust_mnist;

use rand::{rngs::StdRng, Rng, SeedableRng};
use rust_mnist::Mnist;
use std::f64;
use std::time::Instant;

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

        let mut hidden_layers = Vec::with_capacity(hidden_layer_sizes.len());
        for (i, hidden_layer_size) in hidden_layer_sizes.iter().enumerate() {
            if i == 0 {
                hidden_layers.push(new_layer(
                    *hidden_layer_size as usize,
                    input_layer_size as usize,
                ));
            } else {
                hidden_layers.push(new_layer(
                    *hidden_layer_size as usize,
                    hidden_layer_sizes[i - 1] as usize,
                ))
            }
        }

        let output_layer = new_layer(
            output_layer_size as usize,
            hidden_layer_sizes[hidden_layer_sizes.len() - 1] as usize,
        );

        NeuralNetwork {
            input_layer,
            hidden_layers,
            output_layer,
            activation_function,
        }
    }

    fn softmax(&mut self) -> Vec<f64> {
        let mut sum = 0.0;
        for output_neuron in &self.output_layer {
            sum += f64::exp(output_neuron.output);
        }
        let mut softmax_values = Vec::with_capacity(self.output_layer.len());
        for output_neuron in &self.output_layer {
            let softmax_val = f64::exp(output_neuron.output) / sum;
            softmax_values.push(softmax_val);
        }
        return softmax_values;
    }

    fn print_percentages(&mut self) {
        let softmax_values = self.softmax();
        for (i, value) in softmax_values.iter().enumerate() {
            print!("{} -> Percentage: {:.2}%", i, value * 100.0);
            if i != softmax_values.len() - 1 {
                print!(" | ");
            }
        }
        println!();
    }
    fn print_activation(&mut self) {
        for (i, output_neuron) in self.output_layer.iter().enumerate() {
            print!("{} -> Activation: {:.2}", i, output_neuron.output);
            if i != self.output_layer.len() - 1 {
                print!(" | ");
            }
        }

        println!()
    }
    fn compute(&mut self, inputs: &Vec<f64>) {
        // Add input
        for (i, input_neuron) in self.input_layer.iter_mut().enumerate() {
            input_neuron.output = inputs[i];
        }

        for i in 0..self.hidden_layers.len() {
            // Feed foward
            // Hidden layer
            if i == 0 {
                compute_layer(
                    &mut self.hidden_layers[i],
                    &self.input_layer,
                    self.activation_function,
                );
            } else {
                let (previous_layers, current_and_future_layers) =
                    self.hidden_layers.split_at_mut(i);
                compute_layer(
                    &mut current_and_future_layers[0],
                    &previous_layers[i - 1],
                    self.activation_function,
                );
            }
        }
        // Output layer
        compute_layer(
            &mut self.output_layer,
            &self.hidden_layers[self.hidden_layers.len() - 1],
            self.activation_function,
        );
    }
    fn cost(&mut self, datas: &[Data]) -> f64 {
        let mut cost = 0.0;
        for data in datas {
            self.compute(&data.inputs);
            for (i, output_neuron) in self.output_layer.iter().enumerate() {
                cost += (output_neuron.output - output_expected(i as i32, data))
                    * (output_neuron.output - output_expected(i as i32, data));
            }
        }
        return cost / datas.len() as f64;
    }

    fn learn(&mut self, learn_rate: f64, training_data: &[Data]) {
        for data in training_data {
            // Output layer learn
            self.compute(&data.inputs);
            for (output_neuron_index, output_neuron) in self.output_layer.iter_mut().enumerate() {
                let expected = output_expected(output_neuron_index as i32, data);

                output_neuron.delta = 2.0
                    * (output_neuron.output - expected)
                    * (self.activation_function)(output_neuron.weighted_input, true);

                for output_weight_index in 0..output_neuron.weights.len() {
                    let input = self.hidden_layers[self.hidden_layers.len() - 1]
                        [output_weight_index]
                        .output;
                    output_neuron.weights[output_weight_index] -=
                        output_neuron.delta * input * learn_rate;
                }

                output_neuron.bias -= output_neuron.delta * learn_rate;
            }

            // Hidden layer learn
            for layer_index in (0..self.hidden_layers.len()).rev() {
                let mut current_layer = std::mem::take(&mut self.hidden_layers[layer_index]);
                let next_layer = if layer_index == self.hidden_layers.len() - 1 {
                    &self.output_layer
                } else {
                    &self.hidden_layers[layer_index + 1]
                };
                let previous_layer = if layer_index == 0 {
                    &self.input_layer
                } else {
                    &self.hidden_layers[layer_index - 1]
                };

                for current_neuron_index in 0..current_layer.len() {
                    current_layer[current_neuron_index].delta = 0.0;
                    {
                        for next_neuron_index in 0..next_layer.len() {
                            let next_neuron = &next_layer[next_neuron_index];
                            current_layer[current_neuron_index].delta += next_neuron.weights
                                [current_neuron_index]
                                * next_neuron.delta
                                * (self.activation_function)(
                                    current_layer[current_neuron_index].weighted_input,
                                    true,
                                );
                        }

                        for current_weight_index in
                            0..current_layer[current_neuron_index].weights.len()
                        {
                            current_layer[current_neuron_index].weights[current_weight_index] -=
                                current_layer[current_neuron_index].delta
                                    * previous_layer[current_weight_index].output
                                    * learn_rate;
                        }
                    }
                    current_layer[current_neuron_index].bias -=
                        current_layer[current_neuron_index].delta * learn_rate;
                }

                self.hidden_layers[layer_index] = current_layer
            }
        }
    }

    fn train(
        &mut self,
        learn_rate: f64,
        learn_amount: i32,
        epoch_per_learn: i32,
        data_each_epoch: i32,
    ) {
        let training_data = Data::from_minst();
        let mut start_time = Instant::now();
        let mut rng = StdRng::from_entropy();
        let mut rng_index = random_index(&training_data, &mut rng, epoch_per_learn);
        let mut data_slice = &training_data[rng_index..rng_index + data_each_epoch as usize];

        for i in 0..=learn_amount {
            if i % epoch_per_learn == 0 && i != 0 {
                let new_cost = self.cost(data_slice);
                let elapsed = start_time.elapsed().as_secs_f64();
                let speed = data_each_epoch as f64 / elapsed * (epoch_per_learn as f64);
                println!(
                    "Epoch learned {}, cost: {}, elapsed time: {:.2}s, speed: {:.2} Data/s",
                    i, new_cost, elapsed, speed
                );
                start_time = Instant::now();
                rng_index = random_index(&training_data, &mut rng, epoch_per_learn);
                data_slice = &training_data[rng_index..rng_index + data_each_epoch as usize];
            }
            self.learn(learn_rate, data_slice);
        }
    }
}
struct Data {
    inputs: Vec<f64>,
    expected: f64,
}

impl Data {
    fn from_minst() -> Vec<Self> {
        println!("Loading minst...");
        let mut training_data = Vec::new();
        let mnist = Mnist::new("mnist/");
        for (i, mnist_data) in mnist.train_data.iter().enumerate() {
            training_data.push(Data {
                inputs: mnist_data
                    .iter()
                    .map(|&pixel| pixel as f64 / 255.0)
                    .collect(),
                expected: mnist.train_labels[i] as f64,
            })
        }
        println!("Done!");
        return training_data;
    }

    fn add_noise(&mut self, noise_level: f64, probability: f64) {
        let mut rng = StdRng::from_entropy();
        for input in &mut self.inputs {
            if rng.gen::<f64>() > probability {
                *input = max(1.0, *input + rng.gen::<f64>() * noise_level);
            }
        }
    }
}

fn max(a: f64, b: f64) -> f64 {
    if a > b {
        return a;
    }
    return b;
}

fn sigmoid(val: f64, is_derivative: bool) -> f64 {
    let result = 1.0 / (1.0 + f64::exp(-val));
    if is_derivative {
        return result * (1.0 - result);
    }
    return result;
}

fn relu(val: f64, is_derivative: bool) -> f64 {
    if is_derivative {
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
fn random_index(array: &Vec<Data>, rng: &mut StdRng, amount: i32) -> usize {
    let mut rng_index = rng.gen_range(0..array.len());
    while rng_index as i32 + amount >= array.len() as i32 {
        rng_index = rng.gen_range(0..array.len());
    }
    return rng_index;
}
fn new_layer(size: usize, previous_size: usize) -> Vec<Neuron> {
    let mut layer = Vec::with_capacity(size);
    for _ in 0..size {
        layer.push(Neuron {
            delta: 0.0,
            weighted_input: 0.0,
            weights: {
                let mut weights = Vec::with_capacity(previous_size);
                let mut rng = StdRng::from_entropy();
                for _ in 0..previous_size {
                    weights.push(rng.gen_range(-1.0..1.0));
                }
                weights
            },
            bias: 0.0,
            output: 0.0,
        });
    }
    return layer;
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
    const IMAGE_SIZE: usize = 28;
    let mut nn = NeuralNetwork::new(
        vec![100, 16],
        10,
        IMAGE_SIZE as i32 * IMAGE_SIZE as i32,
        sigmoid,
    );

    nn.train(0.03, 10000, 60, 10);
}
