﻿#pragma once

# include <Eigen/Core>
# include <vector>
# include <map>
# include "RNG.h"
# include "Config.h"
# include "Optimizer.h"


class Layer
{
protected:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
	typedef std::map<std::string, int> Meta;

	const int m_in_size;
	const int m_out_size;

public:
	Layer(const int in_size, const int out_size) :
		m_in_size(in_size), m_out_size(out_size) {}

	virtual ~Layer() {}


	/// <summary>
	/// None
	/// </summary>
	/// <returns>Кол-во входящих нейронов</returns>
	int in_size() const
	{
		return m_in_size;
	}

	/// <summary>
	/// None
	/// </summary>
	/// <returns>Кол-во выходящих нейронов</returns>
	int out_size() const
	{
		return m_out_size;
	}

	/// <summary>
	/// Метод инициалиазации слоя.
	/// </summary>
	/// <param name="mu"> - мат.ожидание начальных весов</param>
	/// <param name="sigma"> - дисперсия начальных весов</param>
	/// <param name="rng"> - собственно генератор</param>
	virtual void init(const Scalar& mu, const Scalar& sigma, RNG& rng) = 0;

	/// <summary>
	/// Инициализация слоя без входных параметров, 
	/// будет использоваться при чтении сетки из файла
	/// </summary>
	virtual void init() = 0;

	/// <summary>
	/// Метод прохода вперед по сетке. 
	///  
	/// Будет инициализировано n слоев, у каждого слоя есть такой метод, 
	/// они будут поочередно вызываться из каждого слоя и таким образом, 
	/// мы получим полный проход по всей сетке.
	/// 
	/// Реализация метода сильно отличается у каждого типа слоев
	/// </summary>
	/// <param name="prev_layer_data"> - предыдущий слой, значения его нейронов</param>
	virtual void forward(const Matrix& prev_layer_data) = 0;

	/// <summary>
	/// None
	/// </summary>
	/// <returns>Значения нейронов в слою после функции активации. Использование 
	/// метода разрешено только после метода Layer::forward()!</returns>
	virtual const Matrix& output() = 0;

	/// <summary>
	/// Реализация метода обратного распространения ошибки.
	/// </summary>
	/// <param name="prev_layer_data"> - значения нейронов предыдущего слоя, 
	/// которые также являются входными значениями этого слоя.</param>
	/// <param name="next_layer_data"> - значения нейронов следующего слоя, 
	/// которые также являются выходными значениями этого слоя. </param>
	virtual void backprop(const Matrix& prev_layer_data,
		const Matrix& next_layer_data) = 0;

	/// <summary>
	/// None
	/// </summary>
	/// <returns>Возвращает next_layer_data в Layer::backprop()
	/// </returns>
	virtual const Matrix& backprop_data() const = 0;

	/// <summary>
	/// Обновление параметров сетки после обратного распространения
	/// </summary>
	/// <param name="opt"> - алгоритм оптимизации градиента</param>
	virtual void update(Optimizer& opt) = 0;

	virtual std::vector<Scalar> get_parametrs() const = 0;

	virtual void set_parametrs(const std::vector<Scalar>& param) {};

	virtual std::vector<Scalar> get_derivatives() const = 0;

	virtual std::string layer_type() const = 0;

	virtual std::string activation_type() const = 0;

	/// <summary>
	/// Метод нужен для заполнения основной информации о слое - 
	/// тип слоя, 
	/// входные и выходные значения кол-ва нейронов и т.д.
	/// Нужен для импорта сетки в файл
	/// </summary>
	/// <param name="map"> - словарь</param>
	/// <param name="index"> - индекс слоя в модели</param>
	virtual void fill_meta_info(Meta& map, int index) const = 0;

};
