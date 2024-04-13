use rand::{rngs::StdRng, Rng, SeedableRng};
use std::f64;

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
        for i in 0..hidden_layer_sizes.len() {
            let mut hidden_layer = Vec::with_capacity(hidden_layer_sizes[i] as usize);
            for _ in 0..hidden_layer_sizes[i] {
                hidden_layer.push(Neuron {
                    delta: 0.0,
                    weighted_input: 0.0,
                    weights: if i == 0 {
                        let mut weights = Vec::with_capacity(input_layer_size as usize);
                        for _ in 0..input_layer_size {
                            let mut rng = StdRng::from_entropy();
                            weights.push(rng.gen_range(-1.0..1.0));
                        }
                        weights
                    } else {
                        let previous_layer_size = hidden_layer_sizes[i - 1] as usize;
                        let mut weights = Vec::with_capacity(previous_layer_size as usize);
                        for _ in 0..previous_layer_size {
                            let mut rng = StdRng::from_entropy();
                            weights.push(rng.gen_range(-1.0..1.0));
                        }
                        weights
                    },
                    bias: 0.0,
                    output: 0.0,
                });
            }
            hidden_layers.push(hidden_layer);
        }

        let mut output_layer = Vec::with_capacity(output_layer_size as usize);
        for _ in 0..output_layer_size {
            output_layer.push(Neuron {
                delta: 0.0,
                weighted_input: 0.0,
                weights: {
                    let previous_layer_size =
                        hidden_layer_sizes[hidden_layer_sizes.len() as usize - 1] as usize;
                    let mut weights = Vec::with_capacity(previous_layer_size as usize);
                    for _ in 0..previous_layer_size {
                        let mut rng = StdRng::from_entropy();
                        weights.push(rng.gen_range(-1.0..1.0));
                    }
                    weights
                },
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

        return softmax_values;
    }

    fn print_percentages(&mut self) {
        let softmax_values = self.softmax();
        for i in 0..softmax_values.len() {
            print!("{} -> Percentage: {:.2}%", i, softmax_values[i] * 100.0);
            if i != softmax_values.len() - 1 {
                print!(" | ");
            }
        }
        println!();
    }
    fn print_activation(&mut self) {
        for i in 0..self.output_layer.len() {
            print!("{} -> Activation: {:.2}", i, self.output_layer[i].output);
            if i != self.output_layer.len() - 1 {
                print!(" | ");
            }
        }

        println!()
    }
    fn compute(&mut self, inputs: &Vec<f64>) {
        // Add input
        for i in 0..self.input_layer.len() {
            self.input_layer[i].output = inputs[i];
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
    fn cost(&mut self, datas: &Vec<Data>) -> f64 {
        let mut cost = 0.0;
        for data in datas {
            self.compute(&data.inputs);
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
            // Output layer learn
            self.compute(&data.inputs);
            for i in 0..self.output_layer.len() {
                let neuron = &mut self.output_layer[i];
                let expected = output_expected(i as i32, data);

                neuron.delta = 2.0
                    * (neuron.output - expected)
                    * (self.activation_function)(neuron.weighted_input, true);

                for k in 0..neuron.weights.len() {
                    let input = self.hidden_layers[self.hidden_layers.len() - 1][k].output;
                    neuron.weights[k] -= neuron.delta * input * learn_rate;
                }

                neuron.bias -= neuron.delta * learn_rate;
            }

            // Hidden layer learn
            for i in (0..self.hidden_layers.len()).rev() {
                let mut current_layer = std::mem::take(&mut self.hidden_layers[i]);
                let next_layer = if i == self.hidden_layers.len() - 1 {
                    &self.output_layer
                } else {
                    &self.hidden_layers[i + 1]
                };
                let previous_layer = if i == 0 {
                    &self.input_layer
                } else {
                    &self.hidden_layers[i - 1]
                };

                for j in 0..current_layer.len() {
                    current_layer[j].delta = 0.0;
                    {
                        for k in 0..next_layer.len() {
                            let next_neuron = &next_layer[k];
                            current_layer[j].delta += next_neuron.weights[i]
                                * next_neuron.delta
                                * (self.activation_function)(current_layer[j].weighted_input, true);
                        }

                        for k in 0..current_layer[j].weights.len() {
                            current_layer[j].weights[k] -=
                                current_layer[j].delta * previous_layer[k].output * learn_rate;
                        }
                    }
                    current_layer[j].bias -= current_layer[j].delta * learn_rate;
                }

                self.hidden_layers[i] = current_layer
            }
        }
    }
}
#[derive(Clone)]
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
    let mut nn = NeuralNetwork::new(vec![2], 2, 2, sigmoid);
    let data1 = Data {
        inputs: vec![0.0, 0.0],
        expected: 1.0,
    };
    let data2 = Data {
        inputs: vec![0.0, 1.0],
        expected: 0.0,
    };
    let data3 = Data {
        inputs: vec![1.0, 1.0],
        expected: 0.0,
    };
    let data4 = Data {
        inputs: vec![1.0, 0.0],
        expected: 1.0,
    };
    let mut datas = Vec::new();
    datas.push(data1);
    datas.push(data2);
    datas.push(data3);
    datas.push(data4);
    for i in 0..80 {
        nn.learn(1.5, &datas);
        println!("learned: {}, cost: {}", i, nn.cost(&datas));
    }

    println!("Testing:... 0, 0");
    nn.compute(&datas[0].inputs);
    nn.print_activation();

    // remove not used warning lol
    relu(0.0, true);
    nn.softmax();
    if nn.hidden_layers[0][0].output == 120.0 {
        nn.print_percentages();
    }
}
