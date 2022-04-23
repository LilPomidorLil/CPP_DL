#pragma once

///
/// Этот файл служит как вспомогательная утилита для определения типа слоя, его функции активации и выходного слоя сетки.
/// Все представлено ввиде enum перечисления, что позволит масштабировать эту систему до нужных размеров.
/// 

# include <stdexcept>

namespace internal
{
	enum LAYER_TYPE_ENUM
	{
		FULLYCONNECTED = 0
	};

	/// <summary>
	/// 
	/// </summary>
	/// <param name="type"> - Layers::layer_type()</param>
	/// <returns> - id слоя</returns>
	inline int layer_id(std::string& type)
	{
		if (type == "FullyConnected") return FULLYCONNECTED;

		throw std::invalid_argument("[function layer_id]: unknown type of layer");
		return -1;
	}

	enum ACTIVATION_FUNC_ENUM
	{
		RELU = 0,
		SIGMOID
	};

	/// <summary>
	/// 
	/// </summary>
	/// <param name="type"> - тип функции активации</param>
	/// <returns> - id функции активации</returns>
	inline int activation_id(std::string& type)
	{
		if (type == "ReLU") return RELU;
		if (type == "Sigmoid") return SIGMOID;

		throw std::invalid_argument("[function activation_id]: unknown type of activation func");
		return -1;
	}

	enum OUTPUT_ENUM
	{
		REGRESSIONMSE = 0
	};

	/// <summary>
	/// 
	/// </summary>
	/// <param name="type"> - тип выходного слоя</param>
	/// <returns> - id выходного слоя</returns>
	inline int output_id(std::string& type)
	{
		if (type == "RegressionMSE") return REGRESSIONMSE;

		throw std::invalid_argument("[function output_id]: unknown type of output layer");
		return -1;
	}
}