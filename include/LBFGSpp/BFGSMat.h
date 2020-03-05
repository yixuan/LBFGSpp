// Copyright (C) 2020 Yixuan Qiu <yixuan.qiu@cos.name>
// Under MIT license

#ifndef BFGS_MAT_H
#define BFGS_MAT_H

#include <vector>
#include <Eigen/Core>
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
    typedef std::vector<int> IndexSet;

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

            // Matrix LDLT factorization
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
    // W [n x (2*ncorr)], v [n x 1], res [(2*ncorr) x 1]
    // res preserves the ordering of Y and S columns
    inline void apply_Wtv(const Vector& v, Vector& res) const
    {
        res.resize(2 * m_ncorr);
        for(int j = 0; j < m_ncorr; j++)
        {
            res[j] = m_y.col(j).dot(v);
            res[m_ncorr + j] = m_s.col(j).dot(v);
        }
        res.tail(m_ncorr) *= m_theta;
    }

    // The b-th row of the W matrix
    // Preserves the ordering of Y and S columns
    // Return as a column vector
    inline Vector Wb(int b) const
    {
        Vector res(2 * m_ncorr);
        for(int j = 0; j < m_ncorr; j++)
        {
            res[j] = m_y(b, j);
            res[m_ncorr + j] = m_s(b, j);
        }
        res.tail(m_ncorr) *= m_theta;
        return res;
    }

    // M is [(2*ncorr) x (2*ncorr)], v is [(2*ncorr) x 1]
    inline void apply_Mv(const Vector& v, Vector& res) const
    {
        res.resize(2 * m_ncorr);
        if(m_ncorr < 1)
            return;

        Vector vpadding = Vector::Zero(2 * m_m);
        vpadding.head(m_ncorr).noalias() = v.head(m_ncorr);
        vpadding.segment(m_m, m_ncorr).noalias() = v.tail(m_ncorr);

        // Solve linear equation
        m_permMsolver.solve_inplace(vpadding);

        res.head(m_ncorr).noalias() = vpadding.head(m_ncorr);
        res.tail(m_ncorr).noalias() = vpadding.segment(m_m, m_ncorr);
    }

    // Compute W'Pv
    // W [n x (2*ncorr)], v [nP x 1], res [(2*ncorr) x 1]
    // res preserves the ordering of Y and S columns
    inline void apply_WtPv(const IndexSet& P_set, const Vector& v, Vector& res) const
    {
        const int nP = P_set.size();
        res.resize(2 * m_ncorr);
        if(m_ncorr < 1 || nP < 1)
        {
            res.setZero();
            return;
        }

        for(int j = 0; j < m_ncorr; j++)
        {
            Scalar resy = Scalar(0), ress = Scalar(0);
            for(int i = 0; i < nP; i++)
            {
                const int row = P_set[i];
                resy += m_y(row, j) * v[i];
                ress += m_s(row, j) * v[i];
            }
            res[j] = resy;
            res[m_ncorr + j] = ress;
        }
        res.tail(m_ncorr) *= m_theta;
    }

    // Compute s * P'WMv
    // Assume that v[2*ncorr x 1] has the same ordering (permutation) as W and M
    inline void apply_PtWMv(const IndexSet& P_set, const Vector& v, Vector& res, const Scalar& scale) const
    {
        const int nP = P_set.size();
        res.resize(nP);
        if(m_ncorr < 1 || nP < 1)
        {
            res.setZero();
            return;
        }

        Vector Mv;
        apply_Mv(v, Mv);
        // WP * Mv
        Mv.tail(m_ncorr) *= m_theta;
        for(int i = 0; i < nP; i++)
        {
            Scalar r = Scalar(0);
            const int row = P_set[i];
            for(int j = 0; j < m_ncorr; j++)
            {
                r += m_y(row, j) * Mv[j] + m_s(row, j) * Mv[m_ncorr + j];
            }
            res[i] = r;
        }
        res *= scale;
    }
    // If the P'W matrix has been explicitly formed, do a direct matrix multiplication
    inline void apply_PtWMv(const Matrix& WP, const Vector& v, Vector& res, const Scalar& scale) const
    {
        const int nP = WP.rows();
        res.resize(nP);
        if(m_ncorr < 1 || nP < 1)
        {
            res.setZero();
            return;
        }

        Vector Mv;
        apply_Mv(v, Mv);
        // WP * Mv
        Mv.tail(m_ncorr) *= m_theta;
        res.noalias() = scale * WP * Mv;
    }


    // Compute F'BAb = -(F'W)M(W'AA'd)
    // W'd is known, and AA'+FF'=I, so W'AA'd = W'd - W'FF'd
    // If act_set is smaller, compute W'AA'd = WA' * b
    // If fv_set is smaller, compute W'AA'd = W'd - WF' * (F'd)
    inline void compute_FtBAb(
        const IndexSet& fv_set, const IndexSet& act_set, const Vector& Wd, const Vector& vecd, const Vector& vecb,
        Vector& res
    ) const
    {
        const int nact = act_set.size();
        const int nfree = fv_set.size();
        res.resize(nfree);
        if(m_ncorr < 1 || nact < 1 || nfree < 1)
        {
            res.setZero();
            return;
        }

        // W'AA'd
        Vector rhs(2 * m_ncorr);
        if(nact <= nfree)
        {
            apply_WtPv(act_set, vecb, rhs);
        } else {
            // Construct F'd
            Vector Fd(nfree);
            for(int i = 0; i < nfree; i++)
                Fd[i] = vecd[fv_set[i]];
            // Compute W'AA'd = W'd - WF' * (F'd)
            apply_WtPv(fv_set, Fd, rhs);
            rhs.noalias() = Wd - rhs;
        }

        apply_PtWMv(fv_set, rhs, res, Scalar(-1));
    }

    // Split W in rows
    inline void split_W(const IndexSet& P_set, const IndexSet& L_set, const IndexSet& U_set,
                        Matrix& WP, Matrix& WL, Matrix& WU) const
    {
        const int nP = P_set.size();
        const int nL = L_set.size();
        const int nU = U_set.size();
        WP.resize(nP, 2 * m_ncorr);
        WL.resize(nL, 2 * m_ncorr);
        WU.resize(nU, 2 * m_ncorr);

        // Y part
        for(int j = 0; j < m_ncorr; j++)
        {
            const Scalar* Yptr = &m_y(0, j);
            Scalar* Pptr = WP.data() + j * nP;
            Scalar* Lptr = WL.data() + j * nL;
            Scalar* Uptr = WU.data() + j * nU;
            for(int i = 0; i < nP; i++)
                Pptr[i] = Yptr[P_set[i]];
            for(int i = 0; i < nL; i++)
                Lptr[i] = Yptr[L_set[i]];
            for(int i = 0; i < nU; i++)
                Uptr[i] = Yptr[U_set[i]];
        }
        // S part
        for(int j = 0; j < m_ncorr; j++)
        {
            const Scalar* Sptr = &m_s(0, j);
            Scalar* Pptr = WP.data() + (m_ncorr + j) * nP;
            Scalar* Lptr = WL.data() + (m_ncorr + j) * nL;
            Scalar* Uptr = WU.data() + (m_ncorr + j) * nU;
            for(int i = 0; i < nP; i++)
                Pptr[i] = Sptr[P_set[i]];
            for(int i = 0; i < nL; i++)
                Lptr[i] = Sptr[L_set[i]];
            for(int i = 0; i < nU; i++)
                Uptr[i] = Sptr[U_set[i]];
        }
    }

    // Compute inv(P'BP) * v
    // P represents an index set
    // inv(P'BP) * v = v / theta + WP * inv(inv(M) - WP' * WP / theta) * WP' * v / theta^2
    //
    // v is [nP x 1]
    inline void solve_PtBP(const Matrix& WP, const Vector& v, Vector& res) const
    {
        const int nP = WP.rows();
        res.resize(nP);
        if(m_ncorr < 1 || nP < 1)
        {
            res.noalias() = v / m_theta;
            return;
        }

        // Compute the matrix in the middle (only the lower triangular part is needed)
        // Remember that W = [Y, theta * S], but we do not store theta in WP
        Matrix mid(2 * m_ncorr, 2 * m_ncorr);
        // [0:(ncorr - 1), 0:(ncorr - 1)]
        for(int j = 0; j < m_ncorr; j++)
        {
            for(int i = j; i < m_ncorr; i++)
            {
                mid(i, j) = m_permMinv(i, j) - WP.col(i).dot(WP.col(j)) / m_theta;
            }
        }
        // [ncorr:(2 * ncorr - 1), 0:(ncorr - 1)]
        mid.block(m_ncorr, 0, m_ncorr, m_ncorr).noalias() = m_permMinv.block(m_m, 0, m_ncorr, m_ncorr) -
            WP.rightCols(m_ncorr).transpose() * WP.leftCols(m_ncorr);
        // [ncorr:(2 * ncorr - 1), ncorr:(2 * ncorr - 1)]
        const int offset = m_m - m_ncorr;
        for(int j = m_ncorr; j < 2 * m_ncorr; j++)
        {
            for(int i = j; i < 2 * m_ncorr; i++)
            {
                mid(i, j) = (m_permMinv(offset + i, offset + j) - WP.col(i).dot(WP.col(j))) * m_theta;
            }
        }
        // Factorization
        BKLDLT<Scalar> midsolver(mid);
        // Compute the final result
        Vector WPv = WP.transpose() * v;
        WPv.tail(m_ncorr) *= m_theta;
        midsolver.solve_inplace(WPv);
        WPv.tail(m_ncorr) *= m_theta;
        res.noalias() = v / m_theta + (WP * WPv) / (m_theta * m_theta);
    }

    // Compute P'BQv, where P and Q are two mutually exclusive index selection operators
    // P'BQv = -WP * M * WQ' * v
    inline void apply_PtBQv(const Matrix& WP, const Matrix& WQ, const Vector& v, Vector& res) const
    {
        const int nP = WP.rows();
        const int nQ = WQ.rows();
        res.resize(nP);
        if(m_ncorr < 1 || nP < 1 || nQ < 1)
        {
            res.setZero();
            return;
        }

        // Remember that W = [Y, theta * S], so we need to multiply theta to the second half
        Vector WQtv = WQ.transpose() * v;
        WQtv.tail(m_ncorr) *= m_theta;
        Vector MWQtv;
        apply_Mv(WQtv, MWQtv);
        MWQtv.tail(m_ncorr) *= m_theta;
        res.noalias() = -WP * MWQtv;
    }
};


} // namespace LBFGSpp

/// \endcond

#endif // BFGS_MAT_H
