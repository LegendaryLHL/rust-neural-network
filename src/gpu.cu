extern "C" __global__ void update_weights(double *weights, double delta, double *input, double learn_rate, int len)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < len)
    {
        weights[i] -= delta * input[i] * learn_rate;
    }
}

extern "C" __global__ void learn_output(double **weights, double *delta, double **inputs, double *weighted_input, double *output, double *bias, double learn_rate, int previous_size, int current_size, int data_expected)
{
    // not finished
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < current_size)
    {
        double expected = data_expected == i ? 1.0 : 0.0;
        double activation_derivative = (1 - weighted_input[i]) * weighted_input[i]; // Assuming sigmoid activation function
        delta[i] = 2.0 * (output[i] - expected) * activation_derivative;
        for (int j = 0; j < previous_size; j++)
        {
            weights[i][j] -= delta[i] * inputs[i][j] * learn_rate;
        }
        bias[i] -= delta[i] * learn_rate;
    }
}
