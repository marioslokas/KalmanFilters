using System.Collections;
using System;
using System.IO;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System.Runtime.InteropServices;



namespace AssemblyCSharp
{
	// unscented kalman filter ported from Yi Cao matlab implementation. same outputs as MyKalman class from emgu
	public class MyUnscentedKalman
	{
		#region internal vectordata
		internal class VectorData
		{
			public Emgu.CV.Matrix<double> state;
			public Emgu.CV.Matrix<double> transitionMatrix;
			public Emgu.CV.Matrix<double> measurementMatrix;
			public Emgu.CV.Matrix<double> processNoise;
			public Emgu.CV.Matrix<double> measurementNoise;
			public Emgu.CV.Matrix<double> errorCovariancePost;
			public Emgu.CV.Matrix<double> invMeasurementNoise;
			
			public VectorData()
			{
				// is linear and probably a non linear transition cannot be translated to a transition matrix
				transitionMatrix = new Emgu.CV.Matrix<double>(new double[,] // n * n matrix A, that relates the state at k-1 step to state k step. In practice, it can change at each time step
				                                              {
					//				{1, 0, 0, 1f, 0, 0, 0.5f},  // x-pos, y-pos, z-pos, velocities acceleration expected combination
					//				{0, 1, 0, 0, 1f, 0, 0.5f},
					//				{0, 0, 1, 0, 0, 1f, 0.5f},
					//				{0, 0, 0, 1, 0, 0, 1f},
					//				{0, 0, 0, 0, 1, 0, 1f},
					//				{0, 0, 0, 0, 0, 1, 1f},
					//				{0, 0, 0, 0, 0, 0, 1},
					
//									{1, 0, 0, 1f, 0, 0, 0.5f},  // x-pos, y-pos, z-pos, velocities acceleration expected combination
//									{0, 1, 0, 0, 1f, 0, 0.5f},
//									{0, 0, 1, 0, 0, 1f, 0.5f},
//									{0, 0, 0, 1, 0, 0, 1},
//									{0, 0, 0, 0, 1, 0, 1},
//									{0, 0, 0, 0, 0, 1, 1},
//									{0, 0, 0, 0, 0, 0, 1},
					
					{1, 0, 0, 1f, 0, 0},  // x-pos, y-pos, z-pos, velocities (no accel)
					{0, 1, 0, 0, 1f, 0},
					{0, 0, 1, 0, 0, 1f},
					{0, 0, 0, 1, 0, 0},
					{0, 0, 0, 0, 1, 0},
					{0, 0, 0, 0, 0, 1},
					
				}); 
				
				measurementMatrix = new Emgu.CV.Matrix<double>(new double[,] // m * n matrix H. follows the same rules as transition matrix A
				                                               {
//					                { 1, 0, 0, 0, 0, 0, 0},
//					                { 0, 1, 0, 0, 0, 0, 0},
//					                { 0, 0, 1, 0, 0, 0, 0},
					
					{ 1, 0, 0, 0, 0, 0},
					{ 0, 1, 0, 0, 0, 0},
					{ 0, 0, 1, 0, 0, 0},
				});
				
				measurementMatrix.SetIdentity();
				processNoise = new Emgu.CV.Matrix<double>(6, 6); //Linked to the size of the transition matrix
				/* Q matrix */ processNoise.SetIdentity(new MCvScalar(1.0e-2)); //The smaller the value the more resistance to noise (default e-4)
				measurementNoise = new Emgu.CV.Matrix<double>(3, 3); //Fixed according to input data 
				/* R matrix */measurementNoise.SetIdentity(new MCvScalar(1.0e-2));
				errorCovariancePost = new Emgu.CV.Matrix<double>(6, 6); //Linked to the size of the transition matrix
				errorCovariancePost.SetIdentity();
				invMeasurementNoise = new Emgu.CV.Matrix<double>(3, 3);
			}
			
			public Emgu.CV.Matrix<double> GetMeasurement()
			{
				Emgu.CV.Matrix<double> measurementNoise = new Emgu.CV.Matrix<double>(3, 1);
				measurementNoise.SetRandNormal(new MCvScalar(), new MCvScalar(Math.Sqrt(measurementNoise[0, 0])));
				return measurementMatrix * state + measurementNoise;
			}
			
			public void GoToNextState()
			{
				Emgu.CV.Matrix<double> processNoise = new Emgu.CV.Matrix<double>(6, 1);
				processNoise.SetRandNormal(new MCvScalar(), new MCvScalar(processNoise[0, 0]));
				state = transitionMatrix * state + processNoise;
			}
		}
		
		#endregion
		
		[DllImport(@"Cholesky.dll",
		           EntryPoint = "cholesky_decomposition", CallingConvention = CallingConvention.StdCall)]
		public static extern int Cholesky(double[] source, double[] dest, int size);
		
		#region vars
		Emgu.CV.Matrix<double> sigmaPoints;
		Emgu.CV.Matrix<double> stateCovariance;
		Emgu.CV.Matrix<double> state;
		
		MyUnscentedKalman.VectorData syntheticData;
		
		
		int L; // states
		int m; // measurements
		double c; // scaling factor
		double lambda; // scaling factor
		Emgu.CV.Matrix<double> meansWeights;
		Emgu.CV.Matrix<double> covarianceWeights;
		Emgu.CV.Matrix<double> covarianceWeightsDiagonal;
		
		Emgu.CV.Matrix<double> KalmanGain;
		
		#region tunables
		double alpha = 1e-3;
		double ki = 0; // default 0. has not effect apparently unless the transition is non linear
		double beta = 2; // default 2. same as above
		#endregion
		
		#region transformed vars
		Emgu.CV.Matrix<double> trans_sigmaPoints;
		Emgu.CV.Matrix<double> trans_stateCovariance;
		Emgu.CV.Matrix<double> trans_deviation;
		Emgu.CV.Matrix<double> trans_mean_mat;
		
		Emgu.CV.Matrix<double> trans_cross_covariance;
		#endregion
		
		#endregion
		
		
		// ukf trial from matlab and c++ examples
		// in progress
		
		
		public MyUnscentedKalman (int states, int measurements)
		{
			this.L = states;
			this.m = measurements;
			
			state = new Matrix<double>(this.L, 1);
			state[0,0] = 1;
			state[1,0] = 1;
			state[2,0] = 1;
			state[3,0] = 0.5;
			state[4,0] = 0.5;
			state[5,0] = 0.5;
			state[5,0] = 0.1;

			sigmaPoints = new Matrix<double>(this.L, 2 * this.L + 1);
			stateCovariance = new Matrix<double>(L,L);
			stateCovariance.SetIdentity(new MCvScalar(1.0));
			
			meansWeights = new Matrix<double>(1,2 * this.L + 1);
			covarianceWeights = new Matrix<double>(1,2 * this.L + 1);
			covarianceWeightsDiagonal = new Matrix<double>(2 * this.L + 1,2 * this.L + 1);
			
			calculateVariables();
			
			syntheticData = new VectorData();
			
		}
		
		public Vector3 update(Vector3 point)
		{
			generateSigmaPoints();
			unscentedTransformation(syntheticData.transitionMatrix,sigmaPoints,L,syntheticData.processNoise);
			
			var x1 = trans_mean_mat;
			var x_capital_1 = trans_sigmaPoints;
			var P1 = trans_stateCovariance;
			var x_capital_2 = trans_deviation;
			
			unscentedTransformation(syntheticData.measurementMatrix,x_capital_1,m,syntheticData.measurementNoise);
			
			//updating
			trans_cross_covariance = x_capital_2 * covarianceWeightsDiagonal * trans_deviation.Transpose();
			
			// inverse of P2 (trans_covariance)
			Emgu.CV.Matrix<double> inv_trans_covariance = new Matrix<double>(trans_stateCovariance.Rows,trans_stateCovariance.Cols);
			CvInvoke.cvInvert(trans_stateCovariance,inv_trans_covariance,SOLVE_METHOD.CV_SVD_SYM);
			KalmanGain = trans_cross_covariance * inv_trans_covariance;
			
			Emgu.CV.Matrix<double> thisMeasurement = new Matrix<double>(m,1);
			thisMeasurement[0,0] = point.x;
			thisMeasurement[1,0] = point.y;
			thisMeasurement[2,0] = point.z;
			
			//update state
			state = x1 + KalmanGain * (thisMeasurement - trans_mean_mat);
			
			//update covariance
			stateCovariance = P1 - KalmanGain*trans_cross_covariance.Transpose();

			return new Vector3( (float) state[0,0], (float) state[1,0], (float) state[2,0]);
			
		}
		
		private void unscentedTransformation(Emgu.CV.Matrix<double> map, Emgu.CV.Matrix<double> points, int outputs, Emgu.CV.Matrix<double> additiveCovariance)
		{
			int sigma_point_number = points.Cols; // try points.cols better
			trans_mean_mat = new Matrix<double>(outputs,1);
			trans_sigmaPoints = new Matrix<double>(outputs,sigma_point_number);
			
			for(int i=0; i < sigma_point_number; i++)
			{
				Emgu.CV.Matrix<double> transformed_point = map * points.GetCol(i);
				trans_mean_mat += meansWeights[0,i] * transformed_point;
				
				// store transformed point
				for(int j=0; j < outputs; j++)
				{
					trans_sigmaPoints[j,i] = transformed_point[j,0];
				}
			}

			Emgu.CV.Matrix<double> intermediate_matrix_1 = new Matrix<double>(trans_mean_mat.Rows,sigma_point_number);
			for(int i=0; i < sigma_point_number; i++)
			{
				for(int j=0; j < trans_mean_mat.Rows; j++)
				{
					intermediate_matrix_1[j,i] = trans_mean_mat[j,0];
				}
			}
			
			trans_deviation = trans_sigmaPoints - intermediate_matrix_1; // Y1=Y-y(:,ones(1,L));

			trans_stateCovariance = trans_deviation * covarianceWeightsDiagonal * trans_deviation.Transpose() + additiveCovariance;
			
		}
		
		private void calculateVariables()
		{
			
			lambda = Math.Pow(alpha,2.0) * (L+ki) - L;
			c = L + lambda;
			
			// means weights
			meansWeights[0,0] = (double) (lambda/c);
			
			for(int i=1; i < 2*L+1; i++)
			{
				meansWeights[0,i] = (double) (0.5f/c);
			}
			
			// cov weights
			covarianceWeights = meansWeights.Clone();
			covarianceWeights[0,0] += (double) (1 - Math.Pow(alpha,2.0) + beta); 
			
			// diag of wc
			for(int i=0; i < covarianceWeights.Cols; i++)
			{
				covarianceWeightsDiagonal[i,i] = covarianceWeights[0,i];
			}
			
			c = Math.Sqrt(c);
			
		}
		
		private void generateSigmaPoints()
		{
			Emgu.CV.Matrix<double> A_mat = c * chol(stateCovariance).Transpose();
			Emgu.CV.Matrix<double> Y_mat = new Matrix<double>(state.Rows,state.Rows);
			
			for(int i=0; i < Y_mat.Cols; i++)
			{
				for(int j=0; j < Y_mat.Rows; j++)
				{
					Y_mat[i,j] = state[j,0]; // Y = x(:,ones(1,numel(x)));
				}
			}
			
			// 2 * numel(state) + 1, the reference point
			//first the reference point
			for(int i=0; i < state.Rows; i++)
			{
				sigmaPoints[i,0] = state[i,0];
			}
			
			for(int i=0; i < state.Rows; i++)
			{
				for(int j=0; j < state.Rows; j++)
				{
					sigmaPoints[i,j+1] = Y_mat[j,i] + A_mat[j,i];
				}
			}

			for(int i= 0; i < state.Rows; i++)
			{
				for(int j=0; j < state.Rows; j++)
				{
					sigmaPoints[i,j + state.Rows + 1] = Y_mat[j,i] - A_mat[j,i];
				}
			}
			
			
		}
		
		private Emgu.CV.Matrix<double> chol (Emgu.CV.Matrix<double> input)
		{
			double[] source = new double[input.Rows*input.Cols];
			int i = 0;
			for(int k = 0; k < input.Rows; k++)
			{
				for(int l = 0; l < input.Rows; l++)
				{
					source[i] = input[k,l];
					i++;
				}
			}
			
			double[] destination = new double[input.Rows*input.Cols];
			
			Cholesky(source,destination, input.Rows);
			
			Emgu.CV.Matrix<double> output = new Matrix<double>(input.Rows,input.Cols);
			
			i=0;
			for(int k = 0; k < input.Rows; k++)
			{
				for(int l = 0; l < input.Rows; l++)
				{
					output[k,l] = (double) destination[i];
					i++;
				}
			}
			
			return output;
			
		}
		
		
	}
	
}

