#include <Eigen/Core>
#include <iostream>
#include <LBFGS.h>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using namespace LBFGSpp;

class Rosenbrock
{
private:
    int n;
    ptrdiff_t ncalls;

public:
    Rosenbrock(int n_) : n(n_), ncalls(0) {}
    double operator()(const VectorXd& x, VectorXd& grad)
    {
//        std::cout << x << std::endl;
        ncalls += 1;

        double fx = 0.0;
        for(int i = 0; i < n; i += 2)
        {
            double t1 = 1.0 - x[i];
            double t2 = 10 * (x[i + 1] - x[i] * x[i]);
            grad[i + 1] = 20 * t2;
            grad[i]     = -2.0 * (x[i] * grad[i + 1] + t1);
            fx += t1 * t1 + t2 * t2;
        }
        if (!std::isfinite(fx))
        {
            throw std::runtime_error("fx is not finite");
        }
        return fx;
    }

    const ptrdiff_t get_ncalls() {
      return ncalls;
    }
};

inline void validate_solution(const VectorXd& x)
{
    double diff = (x.array() - 1.0).abs().maxCoeff();
    if (diff > 1e-4)
    {
        throw std::runtime_error("Error is larger than 1e-4");
    }
}

int main()
{
    LBFGSParam<double> param;
    param.    linesearch = LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE;
    param.max_linesearch = 256;

    LBFGSSolver<double, LineSearchBacktracking > solver_backtrack(param);
    LBFGSSolver<double, LineSearchBracketing   > solver_bracket  (param);
    LBFGSSolver<double, LineSearchNocedalWright> solver_nocedal  (param);
    LBFGSSolver<double, LineSearchMoreThuente>   solver_more     (param);

    const int tests_per_n = 1024;

    for( int n=2; n <= 24; n += 2 )
    {
        std::cout << "n = " << n << std::endl;
        Rosenbrock fun_backtrack(n),
                   fun_bracket  (n),
                   fun_nocedal  (n),
                   fun_more     (n);
        int niter_backtrack = 0,
            niter_bracket   = 0,
            niter_nocedal   = 0,
            niter_more      = 0;
        for( int test=0; test < tests_per_n; test++ )
        {
            VectorXd x, x0 = VectorXd::Random(n);

            double fx;

            x = x0; niter_backtrack += solver_backtrack.minimize(fun_backtrack, x, fx); validate_solution(x);
            x = x0; niter_bracket   += solver_bracket  .minimize(fun_bracket  , x, fx); validate_solution(x);
            x = x0; niter_nocedal   += solver_nocedal  .minimize(fun_nocedal  , x, fx); validate_solution(x);
            x = x0; niter_more      += solver_more     .minimize(fun_more     , x, fx); validate_solution(x);
        }
        std::cout << "  Average #calls:" << std::endl;
        std::cout << "  LineSearchBacktracking : " << (fun_backtrack.get_ncalls() / tests_per_n) << " calls, " << (niter_backtrack / tests_per_n) << " iterations" << std::endl;
        std::cout << "  LineSearchBracketing   : " << (fun_bracket  .get_ncalls() / tests_per_n) << " calls, " << (niter_bracket   / tests_per_n) << " iterations" << std::endl;
        std::cout << "  LineSearchNocedalWright: " << (fun_nocedal  .get_ncalls() / tests_per_n) << " calls, " << (niter_nocedal   / tests_per_n) << " iterations" << std::endl;
        std::cout << "  LineSearchMoreThuente  : " << (fun_more     .get_ncalls() / tests_per_n) << " calls, " << (niter_more      / tests_per_n) << " iterations" << std::endl;
    }

    return 0;
}
