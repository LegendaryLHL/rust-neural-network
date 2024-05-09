extern crate rust_mnist;

#[macro_use]
extern crate rustacuda;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rust_mnist::Mnist;
use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;
use std::time::Instant;
use std::{f64, vec};

struct NeuralNetwork {
    delta: Vec<Vec<f64>>,
    weighted_input: Vec<Vec<f64>>,
    bias: Vec<Vec<f64>>,
    output: Vec<Vec<f64>>,
    weights: Vec<Vec<Vec<f64>>>,
    sizes: Vec<i32>,
}
impl NeuralNetwork {
    fn new(layer_sizes: Vec<i32>) -> Self {
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
                        new_weights.push(rng.gen_range(-1.0..1.0));
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
            sizes: layer_sizes,
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

    fn compute_gpu(
        &mut self,
        inputs: &Vec<f64>,
        cuda: &(Module, Stream, Context),
        d_output: &mut DeviceBuffer<f64>,
        d_delta: &mut DeviceBuffer<f64>,
        d_weighted_input: &mut DeviceBuffer<f64>,
        d_bias: &mut DeviceBuffer<f64>,
        d_weights: &mut DeviceBuffer<f64>,
    ) -> Result<(), Box<dyn Error>> {
        let module = &cuda.0;
        let stream = &cuda.1;
        let block_size: u32 = 256;
        let mut d_input = DeviceBuffer::from_slice(inputs)?;
        // add input
        let mut grid_size: u32 = (self.sizes[0] as u32 + block_size - 1) / block_size;
        unsafe {
            launch!(module.add_input<<<grid_size, block_size, 0, stream>>>(
                d_output.as_device_ptr(),
                d_input.as_device_ptr(),
                self.sizes[0]
            ))?;
            stream.synchronize()?;
            // compute layer
            let mut start_index = self.sizes[0];
            let mut weights_start = 0;
            for i in 1..self.sizes.len() {
                grid_size = (self.sizes[i] as u32 + block_size - 1) / block_size;
                launch!(module.compute_layer<<<grid_size, block_size, 0, stream>>>(
                    d_output.as_device_ptr(),
                    d_delta.as_device_ptr(),
                    d_weighted_input.as_device_ptr(),
                    d_bias.as_device_ptr(),
                    d_weights.as_device_ptr(),
                    start_index,
                    weights_start,
                    self.sizes[i],
                    self.sizes[i - 1]
                ))?;
                start_index += self.sizes[i];
                weights_start += self.sizes[i] * self.sizes[i - 1];
                stream.synchronize()?;
            }
        }

        Ok(())
    }
    fn learn(
        &mut self,
        learn_rate: f64,
        training_data: &[Data],
        cuda: &(Module, Stream, Context),
        d_output: &mut DeviceBuffer<f64>,
        d_delta: &mut DeviceBuffer<f64>,
        d_weighted_input: &mut DeviceBuffer<f64>,
        d_bias: &mut DeviceBuffer<f64>,
        d_weights: &mut DeviceBuffer<f64>,
    ) -> Result<(), Box<dyn Error>> {
        let module = &cuda.0;
        let stream = &cuda.1;
        let block_size: u32 = 256;
        for data in training_data {
            self.compute_gpu(
                &data.inputs,
                cuda,
                d_output,
                d_delta,
                d_weighted_input,
                d_bias,
                d_weights,
            )?;

            // Output layer learn
            unsafe {
                let start_index = sum_vec(&self.sizes[0..self.sizes.len() - 1]);
                let weights_start = sum_weights(&self.sizes[0..self.sizes.len() - 1]);
                let grid_size = (*self.sizes.last().unwrap() as u32 + block_size - 1) / block_size;
                launch!(module.learn_output<<<grid_size, block_size, 0, stream>>>(
                    d_output.as_device_ptr(),
                    d_delta.as_device_ptr(),
                    d_weighted_input.as_device_ptr(),
                    d_bias.as_device_ptr(),
                    d_weights.as_device_ptr(),
                    start_index,
                    weights_start,
                    *self.sizes.last().unwrap(),
                    self.sizes[self.sizes.len() - 2],
                    data.expected,
                    learn_rate
                ))?;
                stream.synchronize()?;
            }

            // Hidden layer learn
            for layer_index in (1..self.output.len() - 1).rev() {
                unsafe {
                    let start_index = sum_vec(&self.sizes[0..layer_index]);
                    let weights_start = sum_weights(&self.sizes[0..layer_index]);
                    let grid_size =
                        (*self.sizes.last().unwrap() as u32 + block_size - 1) / block_size;
                    launch!(module.learn_intermediate<<<grid_size, block_size, 0, stream>>>(
                        d_output.as_device_ptr(),
                        d_delta.as_device_ptr(),
                        d_weighted_input.as_device_ptr(),
                        d_bias.as_device_ptr(),
                        d_weights.as_device_ptr(),
                        start_index,
                        weights_start,
                        self.sizes[layer_index],
                        self.sizes[layer_index - 1],
                        self.sizes[layer_index + 1],
                        learn_rate
                    ))?;
                    stream.synchronize()?;
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
        let flattened_weights = flatten_weights(&self.weights);
        let flattened_bias = flatten_vec(&self.bias);
        let mut d_bias = DeviceBuffer::from_slice(&flatten_vec(&self.bias))?;
        let mut d_output = DeviceBuffer::from_slice(&flatten_vec(&self.output))?;
        let mut d_weighted_input = DeviceBuffer::from_slice(&flatten_vec(&self.weighted_input))?;
        let mut d_delta = DeviceBuffer::from_slice(&flatten_vec(&self.delta))?;
        let mut d_weights = DeviceBuffer::from_slice(&flatten_weights(&self.weights))?;
        let training_data = Data::from_minst();
        let mut start_time = Instant::now();
        let mut rng = StdRng::from_entropy();
        let mut rng_index = random_index(&training_data, &mut rng, epoch_per_learn);
        let mut data_slice = &training_data[rng_index..rng_index + data_each_epoch as usize];

        for i in 0..=learn_amount {
            if i % epoch_per_learn == 0 && i != 0 {
                self.copy_bias_from_device(&d_bias, flattened_bias.len())?;
                self.copy_weights_from_device(&d_weights, flattened_weights.len())?;
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
            self.learn(
                learn_rate,
                data_slice,
                &cuda,
                &mut d_output,
                &mut d_delta,
                &mut d_weighted_input,
                &mut d_bias,
                &mut d_weights,
            )?;
        }
        Ok(())
    }

    fn copy_bias_from_device(
        &mut self,
        d_bias: &DeviceBuffer<f64>,
        size: usize,
    ) -> Result<(), Box<dyn Error>> {
        let mut bias_data = vec![0.0; size];
        d_bias.copy_to(&mut bias_data)?;
        let mut start = 0;
        for i in 0..self.bias.len() {
            let end = start + self.bias[i].len();
            self.bias[i] = bias_data[start..end].to_vec();
            start = end;
        }
        Ok(())
    }

    fn copy_weights_from_device(
        &mut self,
        d_weights: &DeviceBuffer<f64>,
        size: usize,
    ) -> Result<(), Box<dyn Error>> {
        let mut weights_data = vec![0.0; size];
        d_weights.copy_to(&mut weights_data)?;

        let mut start = 0;
        for i in 0..self.weights.len() {
            for j in 0..self.weights[i].len() {
                let end = start + self.weights[i][j].len();
                self.weights[i][j] = weights_data[start..end].to_vec();
                start = end;
            }
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
            self.output[layer_index][i] = activation(self.weighted_input[layer_index][i], false);
        }
    }

    fn test_nn(&mut self) -> f64 {
        println!("testing...");
        let test_data = Data::from_minst_test();
        let mut correct_count = 0;
        for data in &test_data {
            self.compute(&data.inputs);
            if self.softmax()[data.expected as usize] * 100.0 > 15.0 {
                correct_count += 1;
            }
        }
        return correct_count as f64 / test_data.len() as f64 * 100.0;
    }
}

struct Data {
    inputs: Vec<f64>,
    expected: i32,
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
                expected: mnist.train_labels[i] as i32,
            })
        }
        println!("Done!");

        return training_data;
    }

    fn from_minst_test() -> Vec<Self> {
        let mut testing_data = Vec::new();
        let mnist = Mnist::new("mnist/");
        for (i, mnist_data) in mnist.test_data.iter().enumerate() {
            testing_data.push(Data {
                inputs: mnist_data
                    .iter()
                    .map(|&pixel| pixel as f64 / 255.0)
                    .collect(),
                expected: mnist.test_labels[i] as i32,
            })
        }

        return testing_data;
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

fn sum_vec(vec: &[i32]) -> i32 {
    let mut sum = 0;
    for value in vec {
        sum += value;
    }
    return sum;
}

fn sum_weights(vec: &[i32]) -> i32 {
    let mut sum = 0;
    for i in 0..vec.len() {
        if i == 0 {
            continue;
        }
        sum += vec[i] * vec[i - 1];
    }
    return sum;
}
fn flatten_vec(vec: &[Vec<f64>]) -> Vec<f64> {
    let mut flattened: Vec<f64> = Vec::new();
    for row in vec {
        flattened.extend_from_slice(&row);
    }
    flattened
}
fn flatten_weights(weights: &[Vec<Vec<f64>>]) -> Vec<f64> {
    let mut flattened: Vec<f64> = Vec::new();
    for layer in weights {
        for row in layer {
            flattened.extend_from_slice(&row);
        }
    }
    flattened
}

fn max(a: f64, b: f64) -> f64 {
    if a > b {
        return a;
    }
    return b;
}

fn activation(val: f64, is_derivative: bool) -> f64 {
    // sigmoid
    let result = 1.0 / (1.0 + f64::exp(-val));
    if is_derivative {
        return result * (1.0 - result);
    }
    return result;
}

fn output_expected(neuron_index: i32, data: &Data) -> f64 {
    if data.expected == neuron_index {
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
    let mut nn = NeuralNetwork::new(vec![IMAGE_SIZE as i32 * IMAGE_SIZE as i32, 100, 16, 10]);
    nn.train(0.03, 5000, 64, 64)?;

    println!("Correct percentage: {:.2}%", nn.test_nn());

    // remove warning
    nn.print_activation();
    nn.print_percentages();
    let mut data = Data {
        inputs: vec![0.1, 0.2],
        expected: 1,
    };
    data.add_noise(0.1, 0.1);
    Ok(())
}
