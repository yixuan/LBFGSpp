#include <Eigen/Core>
#include <iostream>
#include <LBFGS.h>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using namespace LBFGSpp;

double foo(const VectorXd& x, VectorXd& grad)
{
    const int n = x.size();
    VectorXd d(n);
    for(int i = 0; i < n; i++)
        d[i] = i;

    double f = (x - d).squaredNorm();
    grad.noalias() = 2.0 * (x - d);
    return f;
}

int main()
{
    const int n = 10;
    LBFGSParam<double> param;
    LBFGSSolver<double> solver(param);

    VectorXd x = VectorXd::Zero(n);
    solver.minimize(foo, x);

    std::cout << "x = \n" << x << std::endl;

    return 0;
}
