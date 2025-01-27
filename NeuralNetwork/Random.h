﻿#pragma once

# include <Eigen/Core>
# include "RNG.h"
# include "Config.h"

///
/// Здесь собраны утилиты для случайных зависимостей.
/// 



namespace internal
{
    inline void shuffle(int* arr, const int n, RNG& rng)
    {
        for (int i = n - 1; i > 0; i--)
        {
            const int j = int(rng.rand() * (i + 1));

            const int tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
        }
    }

    template <typename DerivedX, typename DerivedY, typename XType, typename YType>
    inline int create_shuffled_batches(
        const Eigen::MatrixBase<DerivedX>& x, const Eigen::MatrixBase<DerivedY>& y,
        int batch_size, RNG& rng,
        std::vector<XType>& x_batches, std::vector<YType>& y_batches
    )
    {
        const int nobs = x.cols();
        const int dimx = x.rows();
        const int dimy = y.rows();

        if (y.cols() != nobs)
        {
            throw std::invalid_argument("Input X and Y have different number of observations");
        }

        Eigen::VectorXi id = Eigen::VectorXi::LinSpaced(nobs, 0, nobs - 1);
        shuffle(id.data(), id.size(), rng);

        if (batch_size > nobs)
        {
            batch_size = nobs;
        }

        const int nbatch = (nobs - 1) / batch_size + 1;
        const int last_batch_size = nobs - (nbatch - 1) * batch_size;
        x_batches.clear();
        y_batches.clear();
        x_batches.reserve(nbatch);
        y_batches.reserve(nbatch);

        for (int i = 0; i < nbatch; i++)
        {
            const int bsize = (i == nbatch - 1) ? last_batch_size : batch_size;
            x_batches.push_back(XType(dimx, bsize));
            y_batches.push_back(YType(dimy, bsize));
            const int offset = i * batch_size;

            for (int j = 0; j < bsize; j++)
            {
                x_batches[i].col(j).noalias() = x.col(id[offset + j]);
                y_batches[i].col(j).noalias() = y.col(id[offset + j]);
            }
        }

        return nbatch;
    }

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