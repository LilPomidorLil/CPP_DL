#pragma once

# include <Eigen/Core>
# include "RNG.h"
# include "Config.h"


namespace internal
{
	inline void set_normal_random(
		Scalar* arr,
		const int n,
		RNG& rng,
		const Scalar& mu = Scalar(0),
		const Scalar& sigma = Scalar(1))
	{
        const double two_pi = 6.283185307179586476925286766559;

        for (int i = 0; i < n - 1; i += 2)
        {
            const double t1 = sigma * std::sqrt(-2 * std::log(rng.rand()));
            const double t2 = two_pi * rng.rand();
            arr[i] = t1 * std::cos(t2) + mu;
            arr[i + 1] = t1 * std::sin(t2) + mu;
        }

        if (n % 2 == 1)
        {
            const double t1 = sigma * std::sqrt(-2 * std::log(rng.rand()));
            const double t2 = two_pi * rng.rand();
            arr[n - 1] = t1 * std::cos(t2) + mu;
        }
	}
}