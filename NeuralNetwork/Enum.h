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
	/// Используя перечисление узнаем какой это тип слоя
	/// </summary>
	/// <param name="type"> - Layers::layer_type()</param>
	/// <returns></returns>
	inline int layer_id(std::string& type)
	{
		if (type == "FullyConnected") return FULLYCONNECTED;

		throw std::invalid_argument("[function layer_id]: unknown type of layer");
		return -1;
	}
}