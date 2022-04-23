#pragma once

# include <Eigen/Core>
# include <iostream>
# include <vector>
# include "Callback.h"
# include "Config.h"
# include "NeuralNetwork.h"


// TODO: Доделать callback до конца.
class MyVerboseCallback : public Callback
{
private:
	/// <summary>
	/// Вычисление среднего лосса по эпохе
	/// </summary>
	/// <param name="loss_arr"> - вектор значений лосса для каждого батча</param>
	Scalar& mean(std::vector<Scalar>& loss_arr)
	{
		Scalar sum = 0.0;
		Scalar mean_loss = 0.0;

		for (int i = 0; i < m_nbatch; ++i)
		{
			sum += loss_arr[i];
		}

		mean_loss = sum / m_nbatch;
		return mean_loss;
	}
public:
	void post_trained_batch(const NeuralNetwork* net,
		const Matrix& x,
		const Matrix& y)
	{
		const Scalar loss = net->get_output()->loss();

		int index = 0;

		std::cout << "[Epoch = " << m_epoch_id << ", batch = " << m_batch_id << "] Loss = " << loss << std::endl;

		if (m_epoch_id - 1 == index)
		{
			index++;
		}
	}

	void post_trained_batch(const NeuralNetwork* net,
		const Matrix& x,
		const IntegerVector& y)
	{
		const Scalar loss = net->get_output()->loss();

		std::cout << "[Epoch = " << m_epoch_id << ", batch = " << m_batch_id << "] Loss = " << loss << std::endl;
	}
};