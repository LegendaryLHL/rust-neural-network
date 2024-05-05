extern crate rust_mnist;
#[macro_use]
extern crate rustacuda;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rust_mnist::Mnist;
use rustacuda::prelude::*;
use std::error::Error;
use std::f64;
use std::ffi::CString;
use std::time::Instant;

struct NeuralNetwork {
    delta: Vec<Vec<f64>>,
    weighted_input: Vec<Vec<f64>>,
    bias: Vec<Vec<f64>>,
    output: Vec<Vec<f64>>,
    weights: Vec<Vec<Vec<f64>>>,
    activation_function: fn(f64, bool) -> f64,
}
impl NeuralNetwork {
    fn new(layer_sizes: Vec<i32>, activation_function: fn(f64, bool) -> f64) -> Self {
        let mut delta = Vec::new();
        let mut weighted_input = Vec::new();
        let mut bias = Vec::new();
        let mut output = Vec::new();
        let mut weights = Vec::new();
        for (i, layer_size) in layer_sizes.iter().enumerate() {
            delta.push(vec![0.0; *layer_size as usize]);
            weighted_input.push(vec![0.0; *layer_size as usize]);
            bias.push(vec![0.0; *layer_size as usize]);
            output.push(vec![0.0; *layer_size as usize]);
            let mut neuron_weights = Vec::new();
            let mut rng = StdRng::from_entropy();
            if i > 0 {
                for _ in 0..*layer_size {
                    let mut new_weights = Vec::new();
                    for _ in 0..layer_sizes[i - 1] {
                        new_weights.push(rng.gen_range(0.0..1.0));
                    }
                    neuron_weights.push(new_weights);
                }
            }
            weights.push(neuron_weights);
        }

        NeuralNetwork {
            delta,
            weighted_input,
            bias,
            output,
            weights,
            activation_function,
        }
    }

    fn softmax(&mut self) -> Vec<f64> {
        let mut sum = 0.0;
        for output in self.output.last().unwrap() {
            sum += f64::exp(*output);
        }
        let mut softmax_values = Vec::with_capacity(self.output.last().unwrap().len());
        for output in self.output.last().unwrap() {
            let softmax_val = f64::exp(*output) / sum;
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
        for (i, output) in self.output.last().unwrap().iter().enumerate() {
            print!("{} -> Activation: {:.2}", i, *output);
            if i != self.output.last().unwrap().len() - 1 {
                print!(" | ");
            }
        }

        println!()
    }
    fn compute(&mut self, inputs: &Vec<f64>) {
        // Add input
        for i in 0..self.output[0].len() {
            self.output[0][i] = inputs[i];
        }

        for i in 1..self.output.len() {
            self.compute_layer(i);
        }
    }

    fn cost(&mut self, datas: &[Data]) -> f64 {
        let mut cost = 0.0;
        for data in datas {
            self.compute(&data.inputs);
            for (i, output) in self.output.last().unwrap().iter().enumerate() {
                cost += (output - output_expected(i as i32, data))
                    * (output - output_expected(i as i32, data));
            }
        }
        return cost / datas.len() as f64;
    }

    fn learn(
        &mut self,
        learn_rate: f64,
        training_data: &[Data],
        cuda: &(Module, Stream, Context),
    ) -> Result<(), Box<dyn Error>> {
        let module = &cuda.0;
        let stream = &cuda.1;
        let block_size: u32 = 256;
        for data in training_data {
            // Output layer learn
            self.compute(&data.inputs);

            for output_neuron_index in 0..self.output.last().unwrap().len() {
                let expected = output_expected(output_neuron_index as i32, data);

                self.delta.last_mut().unwrap()[output_neuron_index] = 2.0
                    * (self.output.last().unwrap()[output_neuron_index] - expected)
                    * (self.activation_function)(
                        self.weighted_input.last().unwrap()[output_neuron_index],
                        true,
                    );

                // Allocate memory on the device and copy data to device
                let mut d_inputs = DeviceBuffer::from_slice(&self.output[self.output.len() - 2])?;
                let mut d_weights = DeviceBuffer::from_slice(
                    &self.weights.last_mut().unwrap()[output_neuron_index],
                )?;

                let grid_size: u32 =
                    (self.weights.last().unwrap().len() as u32 + block_size - 1) / block_size;
                // Call the CUDA function
                unsafe {
                    launch!(module.update_weights<<<grid_size, block_size, 0, stream>>>(
                        d_weights.as_device_ptr(),
                        self.delta.last().unwrap()[output_neuron_index],
                        d_inputs.as_device_ptr(),
                        learn_rate,
                        self.weights.last().unwrap()[output_neuron_index].len()
                    ))?;

                    stream.synchronize()?;
                }

                // Copy the updated weights back to the host
                d_weights.copy_to(&mut self.weights.last_mut().unwrap()[output_neuron_index])?;

                self.bias.last_mut().unwrap()[output_neuron_index] -=
                    self.delta.last().unwrap()[output_neuron_index] * learn_rate;
            }

            // Hidden layer learn
            for layer_index in (1..self.output.len() - 1).rev() {
                for i in 0..self.output[layer_index].len() {
                    self.delta[layer_index][i] = 0.0;

                    for next_neuron_index in 0..self.output[layer_index + 1].len() {
                        self.delta[layer_index][i] += self.weights[layer_index + 1]
                            [next_neuron_index][i]
                            * self.delta[layer_index + 1][next_neuron_index]
                            * (self.activation_function)(self.weighted_input[layer_index][i], true);
                    }

                    // Allocate memory on the device and copy data to device
                    let mut d_inputs = DeviceBuffer::from_slice(&self.output[layer_index - 1])?;
                    let mut d_weights = DeviceBuffer::from_slice(&self.weights[layer_index][i])?;

                    let grid_size: u32 =
                        (self.weights[layer_index][i].len() as u32 + block_size - 1) / block_size;
                    // Call the CUDA function
                    unsafe {
                        launch!(module.update_weights<<<grid_size, block_size, 0, stream>>>(
                            d_weights.as_device_ptr(),
                            self.delta[layer_index][i],
                            d_inputs.as_device_ptr(),
                            learn_rate,
                            self.weights[layer_index][i].len()
                        ))?;

                        stream.synchronize()?;
                    }

                    // Copy the updated weights back to the host
                    d_weights.copy_to(&mut self.weights[layer_index][i])?;

                    self.bias[layer_index][i] -= self.delta[layer_index][i] * learn_rate;
                }
            }
        }
        Ok(())
    }

    fn train(
        &mut self,
        learn_rate: f64,
        learn_amount: i32,
        epoch_per_learn: i32,
        data_each_epoch: i32,
    ) -> Result<(), Box<dyn Error>> {
        let cuda = init_cuda()?;
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
            self.learn(learn_rate, data_slice, &cuda)?;
        }
        Ok(())
    }

    fn compute_layer(&mut self, layer_index: usize) {
        for i in 0..self.output[layer_index].len() {
            self.output[layer_index][i] = 0.0;
            self.weighted_input[layer_index][i] = 0.0;
            for j in 0..self.output[layer_index - 1].len() {
                self.weighted_input[layer_index][i] +=
                    self.output[layer_index - 1][j] * self.weights[layer_index][i][j];
            }
            self.weighted_input[layer_index][i] += self.bias[layer_index][i];
            self.output[layer_index][i] =
                (self.activation_function)(self.weighted_input[layer_index][i], false);
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

fn init_cuda() -> Result<(Module, Stream, Context), Box<dyn Error>> {
    // Initialize the CUDA API
    rustacuda::init(CudaFlags::empty())?;

    // Get the first device
    let device = Device::get_device(0)?;

    // Create a context associated to this device
    let context =
        Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    // Load the module containing the function we want to call
    let module_data = CString::new(include_str!("gpu.ptx"))?;
    let module = Module::load_from_string(&module_data)?;

    // Create a stream to submit work to
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    Ok((module, stream, context))
}

fn main() -> Result<(), Box<dyn Error>> {
    const IMAGE_SIZE: usize = 28;
    let mut nn = NeuralNetwork::new(
        vec![IMAGE_SIZE as i32 * IMAGE_SIZE as i32, 100, 16, 10],
        sigmoid,
    );
    nn.train(0.03, 1000, 10, 10)?;
    Ok(())
}
