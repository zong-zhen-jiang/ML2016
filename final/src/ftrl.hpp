/**
 * ftrl.hpp
 * Implementation of the FTRL-Proximal algorithm
 * Reference paper: http://ppt.cc/nNcA1
 *
 * @author Zong-Zhen Jiang
 * @version 1.0, 12/30/2016
*/

#include <math.h>


using namespace std;


class Ftrl
{
public:

    /**
     * Instantiate a FTRL model with parameters.
     *
     * @param alpha
     * @param beta
     * @param l1
     * @param l2
     * @param vecSize The size of one-hot vector
     */
    Ftrl(double alpha, double beta, double l1, double l2, uint vecSize) :
            mAlpha(alpha),
            mBeta(beta),
            mL1(l1),
            mL2(l2),
            mN(new double[vecSize]()),
            mW(new double[vecSize]()),
            mZ(new double[vecSize]())
    {
    }

    /**
     * Given the feature vector (represented as the indices to the none-zero
     * elements), compute the probability of the ad. being clicked.
     *
     * @param indices Indices to the one-hot vector
     * @param len The length of indices array
     * @return The probability of the ad. being clicked
     */
    double predict(uint * indices, uint len)
    {
        // wx is the inner product of w (weights) and x (feature vector).
        // sigmoid(wx) is the predicted probability.
        double wx = 0.;
        // To understand the for block, please take a look at the pseudocode in
        // the FTRL paper (http://ppt.cc/nNcA1).
        for (uint i = 0; i < len; i++) {
            uint idx = indices[i];
            double sign = (mZ[idx] < 0.)? -1. : 1.;
            if (sign * mZ[idx] <= mL1) {
                mW[idx] = 0.;
            } else {
                mW[idx] = (sign * mL1 - mZ[idx]) /
                        ((mBeta + sqrt(mN[idx])) / mAlpha + mL2);
            }
            wx += mW[idx];
        }
        // We use bounded sigmoid function as the probability estimation.
        return 1. / (1. + exp(-fmax(fmin(wx, 35.), -35.)));
    }

    /**
     * Given the prediced probability using predict(), update the model using
     * ground truth label y.
     *
     * @param indices Indices to the one-hot vector
     * @param len The length of indices array
     * @param p The probability of the ad. being clicked produced by predict()
     * @param y The gound truth label (1 for clicked and 0 otherwise)
     */
    void update(uint * indices, uint len, double p, double y)
    {
        // To understand the for block, again, please take a look at the
        // pseudocode in the FTRL paper (http://ppt.cc/nNcA1).
        double g = p - y;
        for (uint i = 0; i < len; i++) {
            uint idx = indices[i];
            double sigma = (sqrt(mN[idx] + g * g) - sqrt(mN[idx])) / mAlpha;
            mZ[idx] += g - sigma * mW[idx];
            mN[idx] += g * g;
        }
    }


private:

    double mAlpha;
    double mBeta;
    double mL1;
    double mL2;
    double * mN;    // Squared sum of past gradients
    double * mW;    // Lazy weights
    double * mZ;    // Weights
};
