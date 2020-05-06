using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Newtonsoft.Json;
using System.IO;
namespace NeuralNetworkV1
{
    [Serializable]
    public static class Sigmoid
    {
        /// <summary>
        /// Value normalazing function
        /// </summary>
        /// <param name="val">value to normalize</param>
        /// <returns>Normalized value between 0 - 1</returns>
        public static double Activation(double val)
        {
            return (1 / (1 + Math.Pow(Math.E, -val)));
        }

        /// <summary>
        /// Reverse of normalizing
        /// </summary>
        /// <param name="val">Value to denormalize</param>
        /// <returns>original value</returns>
        public static double ActivationDerivative (double val)
        {
            return (Activation(val) * (1 - Activation(val)));
        }
    }
    

    public abstract class Layer
    {
        public bool loggingEnabled = false;
        public double[] nodes;
        public int number;

        public Layer()
        { }
        public Layer(int n)
        {
            this.nodes = new double[n];
            number = n;
        }
        public virtual void Apply()
        {

        }
    }


    [Serializable()]
    public class HiddenLayer : Layer
    {
        public double[,] weights;
        public double[] biases;
        public double[,] weightCorrection;
        public double[] biasCorrecton;

        public HiddenLayer()
        { }

        public HiddenLayer(int n, int x): base(n)
        {
            Random rnd = new Random();            
            this.biases = new double[n];
            this.biasCorrecton = new double[n];
            this.weightCorrection = new double[x,n];
            this.weights = new double[x, n];
            for (int i = 0; i < x; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    this.biases[j] = rnd.NextDouble()*6 - 3;
                    this.weights[i, j] = 0;//rnd.NextDouble()*4 - 2;
                }
            }
        }

        /// <summary>
        /// Computes all neurons 
        /// </summary>
        /// <param name="prev">Previous layer neurons</param>
        public void ComputeNodes(Layer prev)
        {
            StreamWriter writer = new StreamWriter("E:/log.txt", true);
            if (loggingEnabled)
            {               
                writer.WriteLine("---- Layer Compute ----");
            }

            for (int i = 0; i < nodes.Length; i++)
            {
                this.nodes[i] = 0;
                for (int j = 0; j < prev.nodes.Length; j++)
                {
                    nodes[i] += Sigmoid.Activation(prev.nodes[j]) * weights[j, i];
                }
                if(loggingEnabled)
                {
                    writer.WriteLine($"Node№ {i} : value = {nodes[i]} + bias {biases[i]}");
                }
                nodes[i] += biases[i];
                //nodes[i] = Sigmoid.Activation(nodes[i]); // Not needed. Sigmoid activation applied upon next neuron call
            }
            writer.Flush();
            writer.Close();
        }


        /// <summary>
        /// Computes changes by backpropagating errors
        /// </summary>
        /// <param name="errors">Vector of error values</param>
        /// <param name="speed">Speed modifier</param>
        /// <param name="prevLayer">Previous layer neurons</param>
        /// <returns>Errors of previous layer</returns>
        public virtual double[] Propagate(double[] errors, double speed, Layer prevLayer)
        {
            StreamWriter writer = new StreamWriter("E:/log.txt", true);
            if (loggingEnabled)
            {
                writer.WriteLine("---- Layer Error Correction ----");
            }

            double[] prevErrors = new double[prevLayer.number];
            for (int i = 0; i < this.number; i++)
            {
                for (int j = 0; j < prevLayer.number; j++)
                {
                   prevErrors[i] += errors[i] * weights[j, i];
                }

                if (loggingEnabled)
                {
                    writer.WriteLine($"Eror {i} : value = {prevErrors[i]}");
                }
                prevErrors[i] = Sigmoid.ActivationDerivative(prevErrors[i]);
                if (loggingEnabled)
                {
                    writer.WriteLine($"Error deactivated {i} : value = {prevErrors[i]}");
                }

                for (int j = 0; j < prevLayer.number; j++)
                {
                    weightCorrection[j, i] = speed * prevErrors[i] * Sigmoid.Activation(prevLayer.nodes[j]);
                    if (loggingEnabled)
                    {
                        //writer.WriteLine($"Weight correction [{i},{j}] : value = { weightCorrection[j, i]}");
                    }
                }
                biasCorrecton[i] = prevErrors[i] * speed;
                if (loggingEnabled)
                {
                    writer.WriteLine($"Bias correction {i} : value = {biasCorrecton[i]}");
                }
            }
            writer.Flush();
            writer.Close();
            return (prevErrors);
        }

        /// <summary>
        /// Applies changes to weights and biases
        /// </summary>
        public void Apply()
        {
            for (int i = 0; i < number; i++)
            {
                for (int j = 0; j < weights.GetLength(0); j++)
                {
                    weights[j, i] += weightCorrection[j, i];
                }
                biases[i] = biasCorrecton[i];
            }
        }

    }



    [Serializable()]
    public class OutputLayer : HiddenLayer
    {
        public OutputLayer()
        {
        }
        public OutputLayer(int n, int x) : base(n, x)
        {
        }

        /// <summary>
        /// Computes changes by backpropagating errors.
        /// </summary>
        /// <param name="errors">Vector of error values</param>
        /// <param name="speed">Speed modifier</param>
        /// <param name="prevLayer">Previous layer neurons</param>
        /// <returns>Errors of previous layer</returns>
        public override double[] Propagate(double[] errors, double speed, Layer prevLayer)
        {
            double[] prevErrors = new double[prevLayer.number];
            for (int i = 0; i < this.number; i++)
            {
                for (int j = 0; j < prevLayer.number; j++)
                {
                    prevErrors[i] += errors[i] * weights[j, i];
                }

                for (int j = 0; j < prevLayer.number; j++)
                {
                    weightCorrection[j, i] = speed * errors[i] * Sigmoid.Activation(prevLayer.nodes[j]);
                }
                biasCorrecton[i] =  speed * errors[i];
            }
            return (errors);
        }

        /*
        public override double[] Propagate(double[] errors, double speed, Layer prevLayer)
        {
            double[] prevErrors = new double[prevLayer.number];
            for (int i = 0; i < this.number; i++)
            {
                for (int j = 0; j < prevLayer.number; j++)
                {
                    weightCorrection[i,j] = speed * errors[i] * prevLayer.nodes[j];
                    prevErrors[j] += errors[i] * weights[i, j];
                }
                biasCorrecton[i] = speed * errors[i];
            }
            return (prevErrors);
        }
        */
    }


    [Serializable()]
    public class InputLayer : Layer
    {
        public InputLayer()
        { }

        public InputLayer(int n) : base(n)
        { }

    }



    [Serializable()]
    public class NeuralNet
    {
        public double speed { get; set; }
        public InputLayer inputLayer { get; set; }
        public List<HiddenLayer> hiddenLayers { get; set; }
        public HiddenLayer outputLayer { get; set; }

        public NeuralNet()
        {
        }
        public NeuralNet(int inputNodes,int hiddenLayers, int hiddenNodes, int outputNodes)
        {
            speed = 0.1;
            inputLayer = new InputLayer(inputNodes);
            this.hiddenLayers = new List<HiddenLayer>();
            this.hiddenLayers.Add( new HiddenLayer(hiddenNodes, inputNodes));
            for (int i = 1; i < hiddenLayers; i++)
            { 
                this.hiddenLayers.Add(new HiddenLayer(hiddenNodes, hiddenNodes));
            }
            outputLayer = new HiddenLayer(outputNodes, hiddenNodes);
        }



        /// <summary>
        /// Computes all neurons in all layers.
        /// </summary>
        /// <param name="input">Input vector</param>
        /// <returns>Vector of probabilities</returns>
        public double[] Compute(double[] input)
        {
            double[] output = new double[outputLayer.number];
            inputLayer.nodes = input;
            hiddenLayers[0].ComputeNodes(inputLayer);
            for (int i = 1; i < hiddenLayers.Count; i++)
            {
                hiddenLayers[i].ComputeNodes(hiddenLayers[i - 1]);
            }
            outputLayer.ComputeNodes(hiddenLayers[hiddenLayers.Count - 1]);
            for (int i = 0; i < outputLayer.number; i++)
            {
                output[i] = Sigmoid.Activation(outputLayer.nodes[i]);
            }
            return (output);
        }

        /// <summary>
        /// Computes all neurons in all layers, propagetes error.
        /// </summary>
        /// <param name="input">Input vector</param>
        /// <param name="expected">Vector of expected result probabilities</param>
        /// <returns>Vector of probabilities</returns>
        public double[] Learn(double[] input, double[] expected)
        {
            double[] result = this.Compute(input);
            double[] errors = new double[outputLayer.number];
            for (int i = 0; i < result.Length; i++)
            {
                errors[i] = (expected[i] - result[i])*Sigmoid.ActivationDerivative(outputLayer.nodes[i]);                
            }
            double[] prevErrors = outputLayer.Propagate(errors, speed, hiddenLayers.Last());
            prevErrors = hiddenLayers.Last().Propagate(errors, speed, outputLayer);
            for (int i = hiddenLayers.Count-1; i >= 0; i--)
            {
                prevErrors = hiddenLayers[i].Propagate(prevErrors, speed, hiddenLayers[i + 1]);
            }
           
            foreach (var item in hiddenLayers)
            {
                item.Apply();
            }
            outputLayer.Apply();
            return result;
        }

    }
}
