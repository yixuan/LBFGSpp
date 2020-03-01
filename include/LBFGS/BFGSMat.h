// Copyright (C) 2020 Yixuan Qiu <yixuan.qiu@cos.name>
// Under MIT license

#ifndef BFGS_MAT_H
#define BFGS_MAT_H

#include <vector>
#include <Eigen/Core>
#include <Eigen/LU>
#include "BKLDLT.h"


/// \cond

namespace LBFGSpp {


//
// An *implicit* representation of the BFGS approximation to the Hessian matrix B
//
// B = theta * I - W * M * W'
// H = inv(B)
//
// Reference:
// [1] D. C. Liu and J. Nocedal (1989). On the limited memory BFGS method for large scale optimization.
// [2] R. H. Byrd, P. Lu, and J. Nocedal (1995). A limited memory algorithm for bound constrained optimization.
//
template <typename Scalar, bool LBFGSB = false>
class BFGSMat
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Ref<const Vector> RefConstVec;
    typedef std::vector<int> IntVector;

    int    m_m;      // Maximum number of correction vectors
    Scalar m_theta;  // theta * I is the initial approximation to the Hessian matrix
    Matrix m_s;      // History of the s vectors
    Matrix m_y;      // History of the y vectors
    Vector m_ys;     // History of the s'y values
    Vector m_alpha;  // Temporary values used in computing H * v
    int    m_ncorr;  // Number of correction vectors in the history, m_ncorr <= m
    int    m_ptr;    // A Pointer to locate the most recent history, 1 <= m_ptr <= m
                     // Details: s and y vectors are stored in cyclic order.
                     //          For example, if the current s-vector is stored in m_s[, m-1],
                     //          then in the next iteration m_s[, 0] will be overwritten.
                     //          m_s[, m_ptr-1] points to the most recent history,
                     //          and m_s[, m_ptr % m] points to the most distant one.

    //========== The following members are only used in L-BFGS-B algorithm ==========//
    Matrix                      m_Minv;         // M inverse
    Eigen::PartialPivLU<Matrix> m_Msolver;      // Represents the M matrix, since Minv is easy to form
    Matrix                      m_permMinv;     // Permutated M inverse
    BKLDLT<Scalar>              m_permMsolver;  // Represents the permutated M matrix

public:
    // Constructor
    BFGSMat() {}

    // Reset internal variables
    // n: dimension of the vector to be optimized
    // m: maximum number of corrections to approximate the Hessian matrix
    inline void reset(int n, int m)
    {
        m_m = m;
        m_theta = Scalar(1);
        m_s.resize(n, m);
        m_y.resize(n, m);
        m_ys.resize(m);
        m_alpha.resize(m);
        m_ncorr = 0;
        m_ptr = m;  // This makes sure that m_ptr % m == 0 in the first step

        if(LBFGSB)
        {
            m_permMinv.resize(2 * m, 2 * m);
            m_permMinv.diagonal().setOnes();
        }
    }

    // Add correction vectors to the BFGS matrix
    inline void add_correction(const RefConstVec& s, const RefConstVec& y)
    {
        const int loc = m_ptr % m_m;

        m_s.col(loc).noalias() = s;
        m_y.col(loc).noalias() = y;

        // ys = y's = 1/rho
        const Scalar ys = m_s.col(loc).dot(m_y.col(loc));
        m_ys[loc] = ys;

        m_theta = m_y.col(loc).squaredNorm() / ys;

        if(m_ncorr < m_m)
            m_ncorr++;

        m_ptr = loc + 1;

        if(LBFGSB)
        {
            // Minv = [-D         L']
            //        [ L  theta*S'S]

            // Copy -D
            // Let S=[s[0], ..., s[m-1]], Y=[y[0], ..., y[m-1]]
            // D = [s[0]'y[0], ..., s[m-1]'y[m-1]]
            m_permMinv(loc, loc) = -ys;

            // Update S'S
            // We only store S'S in Minv, and multiply theta when LU decomposition is performed
            Vector Ss = m_s.leftCols(m_ncorr).transpose() * m_s.col(loc);
            m_permMinv.block(m_m + loc, m_m, 1, m_ncorr).noalias() = Ss.transpose();
            m_permMinv.block(m_m, m_m + loc, m_ncorr, 1).noalias() = Ss;

            // Compute L
            // L = [          0                                     ]
            //     [  s[1]'y[0]             0                       ]
            //     [  s[2]'y[0]     s[2]'y[1]                       ]
            //     ...
            //     [s[m-1]'y[0] ... ... ... ... ... s[m-1]'y[m-2]  0]
            //
            // L_next = [        0                                   ]
            //          [s[2]'y[1]             0                     ]
            //          [s[3]'y[1]     s[3]'y[2]                     ]
            //          ...
            //          [s[m]'y[1] ... ... ... ... ... s[m]'y[m-1]  0]
            const int len = m_ncorr - 1;
            // First zero out the column of oldest y
            if(m_ncorr >= m_m)
                m_permMinv.block(m_m, loc, m_m, 1).setZero();
            // Compute the row associated with new s
            // The current row is loc
            // End with column (loc + m - 1) % m
            // Length is len
            int yloc = (loc + m_m - 1) % m_m;
            for(int i = 0; i < len; i++)
            {
                m_permMinv(m_m + loc, yloc) = m_s.col(loc).dot(m_y.col(yloc));
                yloc = (yloc + m_m - 1) % m_m;
            }

            m_permMinv.block(m_m, m_m, m_m, m_m) *= m_theta;
            m_permMsolver.compute(m_permMinv);
            m_permMinv.block(m_m, m_m, m_m, m_m) /= m_theta;
        }
    }

    // Recursive formula to compute a * H * v, where a is a scalar, and v is [n x 1]
    // H0 = (1/theta) * I is the initial approximation to H
    // Algorithm 7.4 of Nocedal, J., & Wright, S. (2006). Numerical optimization.
    inline void apply_Hv(const Vector& v, const Scalar& a, Vector& res)
    {
        res.resize(v.size());

        // L-BFGS two-loop recursion

        // Loop 1
        res.noalias() = a * v;
        int j = m_ptr % m_m;
        for(int i = 0; i < m_ncorr; i++)
        {
            j = (j + m_m - 1) % m_m;
            m_alpha[j] = m_s.col(j).dot(res) / m_ys[j];
            res.noalias() -= m_alpha[j] * m_y.col(j);
        }

        // Apply initial H0
        res /= m_theta;

        // Loop 2
        for(int i = 0; i < m_ncorr; i++)
        {
            const Scalar beta = m_y.col(j).dot(res) / m_ys[j];
            res.noalias() += (m_alpha[j] - beta) * m_s.col(j);
            j = (j + 1) % m_m;
        }
    }

    //========== The following functions are only used in L-BFGS-B algorithm ==========//

    // Return the value of theta
    inline Scalar theta() const { return m_theta; }

    // Return current number of correction vectors
    inline int num_corrections() const { return m_ncorr; }

    // W = [Y, theta * S]
    // W is [n x (2*ncorr)], v is [n x 1]
    inline void apply_Wtv(const Vector& v, Vector& res) const
    {
        res.resize(2 * m_ncorr);

        // To iterate from the most recent history to most distant,
        // j = ptr - 1
        // for i = 0, ..., ncorr
        //     j points to the column of y or s that is i away from the most recent one
        //     j = (j + m - 1) % m
        // end for

        // Pointer to most recent history
        int j = m_ptr - 1;
        for(int i = 0; i < m_ncorr; i++)
        {
            // res[0] corresponds to the oldest y
            // res[ncorr-1]   => newest y
            // res[ncorr]     => oldest s
            // res[2*ncorr-1] => newest s
            res[m_ncorr - 1 - i] = m_y.col(j).dot(v);
            res[2 * m_ncorr - 1 - i] = m_theta * m_s.col(j).dot(v);
            j = (j + m_m - 1) % m_m;
        }
    }

    // The b-th row of the W matrix
    // Return as a column vector
    inline Vector Wb(int b) const
    {
        Vector res(2 * m_ncorr);
        int j = m_ptr - 1;
        for(int i = 0; i < m_ncorr; i++)
        {
            res[m_ncorr - 1 - i] = m_y(b, j);
            res[2 * m_ncorr - 1 - i] = m_theta * m_s(b, j);
            j = (j + m_m - 1) % m_m;
        }
        return res;
    }

    // Return some rows of W based on the index set b
    inline Matrix Wb(const IntVector& b) const
    {
        const int nb = b.size();
        Matrix res(nb, 2 * m_ncorr);

        int j = m_ptr - 1;
        for(int i = 0; i < m_ncorr; i++)
        {
            for(int k = 0; k < nb; k++)
            {
                res(k, m_ncorr - 1 - i) = m_y(b[k], j);
                res(k, 2 * m_ncorr - 1 - i) = m_theta * m_s(b[k], j);
            }
            j = (j + m_m - 1) % m_m;
        }
        return res;
    }

    // Compute the inverse M matrix and the associated factorization
    inline void form_M()
    {
        if(m_ncorr < 1)
            return;

        // Minv = [-D         L']
        //        [ L  theta*S'S]
        m_Minv.resize(2 * m_ncorr, 2 * m_ncorr);
        m_Minv.setZero();
        // Pointer to most recent history
        const int loc = m_ptr - 1;
        // Segment 1: 0, 1, ..., loc
        // Copy to ncorr-len1, ..., ncorr-1
        const int len1 = loc + 1;
        // Segment 2: loc+1, ..., m-1, only when ncorr == m
        // Copy to 0, 1, ..., len2-1
        const int len2 = m_m - loc - 1;

        // Copy -D
        // To iterate from the most recent history to most distant,
        // j = ptr - 1
        // for i = 0, ..., ncorr
        //     j points to the column of y or s that is i away from the most recent one
        //     j = (j + m - 1) % m
        // end for
        int i = 0, j = m_ptr - 1;
        for(i = 0; i < m_ncorr; i++)
        {
            m_Minv(m_ncorr - i - 1, m_ncorr - i - 1) = -m_ys[j];
            j = (j + m_m - 1) % m_m;
        }

        // Compute L
        // Let S=[s[0], ..., s[m-1]], Y=[y[0], ..., y[m-1]]
        // L = [          0                                     ]
        //     [  s[1]'y[0]             0                       ]
        //     [  s[2]'y[0]     s[2]'y[1]                       ]
        //     ...
        //     [s[m-1]'y[0] ... ... ... ... ... s[m-1]'y[m-2]  0]
        //
        // We only use y up to y[m-2]
        int jloc = m_ptr - 1;
        jloc = (jloc + m_m - 1) % m_m;
        for(j = m_ncorr - 2; j >= 0; j--)
        {
            int iloc = m_ptr - 1;
            for(i = m_ncorr - 1; i > j; i--)
            {
                m_Minv(m_ncorr + i, j) = m_y.col(jloc).dot(m_s.col(iloc));
                iloc = (iloc + m_m - 1) % m_m;
            }
            jloc = (jloc + m_m - 1) % m_m;
        }
        // Copy to the top right corner
        m_Minv.topRightCorner(m_ncorr, m_ncorr).noalias() = m_Minv.bottomLeftCorner(m_ncorr, m_ncorr).transpose();

        // Compute theta*S'S
        m_Minv.bottomRightCorner(len1, len1).noalias() = m_theta * m_s.leftCols(len1).transpose() * m_s.leftCols(len1);
        if(m_ncorr == m_m && len2 > 0)
        {
            m_Minv.block(m_ncorr, m_ncorr, len2, len2).noalias() = m_theta * m_s.rightCols(len2).transpose() * m_s.rightCols(len2);
            m_Minv.block(m_ncorr + len2, m_ncorr, len1, len2).noalias() = m_theta * m_s.leftCols(len1).transpose() * m_s.rightCols(len2);
            m_Minv.block(m_ncorr, m_ncorr + len2, len2, len1).noalias() = m_Minv.block(m_ncorr + len2, m_ncorr, len1, len2).transpose();
        }

        m_Msolver.compute(m_Minv);
    }

    inline const Matrix& Minv() const { return m_Minv; }

    // M is [(2*ncorr) x (2*ncorr)], v is [(2*ncorr) x 1]
    inline void apply_Mv(const Vector& v, Vector& res) const
    {
        res.resize(2 * m_ncorr);
        if(m_ncorr < 1)
            return;

        // Permute v to align with M
        Vector vperm = Vector::Zero(2 * m_m);
        int loc = m_ptr - 1;
        // From newest to oldest
        for(int i = 0; i < m_ncorr; i++)
        {
            vperm[loc] = v[m_ncorr - i - 1];
            vperm[m_m + loc] = v[2 * m_ncorr - i - 1];
            loc = (loc + m_m - 1) % m_m;
        }

        // Solve linear equation
        m_permMsolver.solve_inplace(vperm);

        // Permute the solution back
        // From newest to oldest
        loc = m_ptr - 1;
        for(int i = 0; i < m_ncorr; i++)
        {
            res[m_ncorr - i - 1] = vperm[loc];
            res[2 * m_ncorr - i - 1] = vperm[m_m + loc];
            loc = (loc + m_m - 1) % m_m;
        }
    }

    // Compute P'BQv, where P and Q are two index selection operators
    inline void apply_PtBQv(const IntVector& P_set, const IntVector& Q_set, const Vector& v, Vector& res) const
    {
        res.resize(P_set.size());
        if(m_ncorr < 1 || P_set.size() < 1 || Q_set.size() < 1)
        {
            res.setZero();
            return;
        }

        Matrix WP = Wb(P_set), WQ = Wb(Q_set);
        Vector WQtv = WQ.transpose() * v;
        Vector MWQtv;
        apply_Mv(WQtv, MWQtv);
        res.noalias() = -WP * MWQtv;
    }
};


} // namespace LBFGSpp

/// \endcond

#endif // BFGS_MAT_H
