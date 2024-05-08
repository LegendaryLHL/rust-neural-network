#include <math.h>
#include <stdio.h>
extern "C" __global__ void update_weights(double *weights, double delta, double *input, double learn_rate, int len)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < len)
    {
        weights[i] -= delta * input[i] * learn_rate;
    }
}
extern "C" __global__ void add_input(double *output, double *input, int len)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < len)
    {
        output[i] = input[i];
    }
}
extern "C" __global__ void compute_layer(double *output, double *delta, double *weighted_input, double *bias, double *weights, int start_index, int weights_start, int len, int previous_len)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int previous_start = start_index - previous_len;
    if (i < len)
    {
        int index = i + start_index;
        output[index] = 0.0;
        weighted_input[index] = 0.0;
        for (int j = 0; j < previous_len; j++)
        {
            weighted_input[index] += output[previous_start + j] * weights[weights_start + i * previous_len + j];
        }

        weighted_input[index] += bias[index];
        // sigmoid
        output[index] = 1.0 / (1.0 + exp(-weighted_input[index]));
    }
}

extern "C" __global__ void learn_output(double *output, double *delta, double *weighted_input, double *bias, double *weights, int start_index, int weights_start, int len, int previous_len, int expected_index, double learn_rate)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int previous_start = start_index - previous_len;
    if (i < len)
    {
        int index = i + start_index;
        double expected = expected_index == i ? 1.0 : 0.0;

        // Sigmoid activation function derivative
        double weighted_input_derivative = 1.0 / (1.0 + exp(-weighted_input[index]));
        double activation_derivative = (1 - weighted_input_derivative) * weighted_input_derivative;
        delta[index] = 2.0 * (output[index] - expected) * activation_derivative;
        for (int j = 0; j < previous_len; j++)
        {
            weights[weights_start + i * previous_len + j] -= delta[index] * output[previous_start + j] * learn_rate;
        }
        bias[index] -= delta[index] * learn_rate;
    }
}

extern "C" __global__ void learn_intermediate(double *output, double *delta, double *weighted_input, double *bias, double *weights, int start_index, int weights_start, int len, int previous_len, int next_len, double learn_rate)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int previous_start = start_index - previous_len;
    int next_start = start_index + len;
    int next_weights_start = weights_start + len * previous_len;
    if (i < len)
    {
        int index = i + start_index;
        delta[index] = 0.0;
        double weighted_input_derivative = 1.0 / (1.0 + exp(-weighted_input[index]));
        double activation_derivative = (1 - weighted_input_derivative) * weighted_input_derivative;

        for (int j = 0; j < next_len; j++)
        {
            delta[index] += delta[next_start + j] * activation_derivative * weights[next_weights_start + i * len + j];
        }

        for (int j = 0; j < previous_len; j++)
        {
            weights[weights_start + i * previous_len + j] -= delta[index] * output[previous_start + j] * learn_rate;
        }
        bias[index] -= delta[index] * learn_rate;
    }
}
