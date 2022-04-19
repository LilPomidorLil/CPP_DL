#pragma once

# include <Eigen/Core>
# include <map>
# include <vector>
# include <stdexcept>
# include "Config.h"
# include "RNG.h"
# include "Layer.h"
# include "Output.h"
# include "Callback.h"


///
/// Этот модуль описывает интерфейс нейронной сети, которая будет использоваться пользователем
/// 


class NeuralNetwork
{
private:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef Eigen::RowVectorXi IntegerVector;
	typedef std::map<std::string, int> Meta;


	RNG m_default_rng; // дефолтный генератор
	RNG& m_rng; // генератор, преданный пользователем (ссылка на генератор), иначе дефолт

	std::vector<Layer*> m_layers; // указатели на созданные пользователем слои сетки
	Output* m_output; // указатель на выходной слой
	Callback m_default_callback; // дефолтный вывод на печать
	Callback* m_callback; // пользовательский вывод на печать, иначе дефолт

	/// <summary>
	/// Проверка всех слоев на соотвествие вход текущего == выход предыдущего
	/// </summary>
	void check_unit_sizes() const
	{
		const int nlayer = count_layers();

		if (nlayer <= 1) { return; }

		for (int i = 1; i < nlayer; ++i)
		{
			if (m_layers[i]->in_size() != m_layers[i - 1]->out_size())
			{
				throw std::invalid_argument("[class NeuralNetwork]: Unit sizes do not match");
			}
		}
	}

	/// <summary>
	/// Проход по всей сетке.
	/// </summary>
	/// <param name="input"> - входные данные. 
	/// Убедитесь, что их длина равна длине входного слоя сетки</param>
	void forward(const Matrix& input)
	{
		const int nlayer = count_layers();

		if (nlayer <= 0) { return; }

		/// Проверим нулевой слой на соотвествие правилу вход данных == вход нулевого слоя

		if (input.rows() != m_layers[0]->in_size())
		{
			throw std::invalid_argument("[class NeuralNetwork]: Input data have incorrect dimension");
		}

		// Протолкнули данные в нулевой слой

		m_layers[0]->forward(input);

		// Начинаем толкать данные по всей сетке

		for (int i = 1; i < nlayer; ++i)
		{
			m_layers[i]->forward(m_layers[i - 1]->output());
		}

		// На этом проход по всей сетке завершен
	}

	/// <summary>
	/// Backprop для всей сетки. Просто идем из конца сетки в ее начало и вызываем
	/// у каждого слоя метод обратного распространения
	/// </summary>
	/// <typeparam name="TargetType"> - тип таргета, отедльно для задач бинарной
	/// классификации и регрессии, а также для многоклассовой классификации</typeparam>
	/// <param name="input"> - входные данные</param>
	/// <param name="target"> - собственно таргет</param>
	template <typename TargetType>
	void backprop(const Matrix& input, const TargetType& target)
	{
		const int nlayer = count_layers();

		if (nlayer <= 0) { return; }

		// Создадим указатель на первый и последний (скрытый, но не выходной)
		// слой сетки, это поможет в дальнейшем

		Layer* first_layer = m_layers[0];
		Layer* last_layer = m_layers[nlayer - 1];

		// Начнем распространение с конца сетки
		m_output->check_target_data(target);
		m_output->evaluate(last_layer->output(), target);

		// Если скрытый слой всего один, то 'prev_layer_data' будут выходными данными
		
		if (nlayer == 1)
		{
			first_layer->backprop(input, last_layer->backprop_data());
			return;
		}

		// Если это условие не выполнено, то вычисляем градиент для последнего скрытого слоя
		last_layer->backprop(m_layers[nlayer - 2]->output(), m_output->backprop_data());

		// Теперь пробегаемся по всем слоям и вычисляем градиенты

		for (int i = nlayer - 2; i > 0; --i)
		{
			m_layers[i]->backprop(m_layers[i - 1]->output(),
							      m_layers[i + 1]->backprop_data());
		}

		// Теперь вычисляем грады для нулевого - входного слоя сетки

		first_layer->backprop(input, m_layers[1]->backprop_data());

		// На этом backprop окончен
	}

	/// <summary>
	/// Обновление весов модели
	/// </summary>
	/// <param name="opt"> - собственно оптимайзер</param>
	void update(Optimizer& opt)
	{
		const int nlayer = count_layers();

		if (nlayer <= 0) { return; }

		for (int i = 0; i < nlayer; ++i)
		{
			m_layers[i]->update(opt);
		}
	}

	Meta get_meta_info() const
	{
		const int nlayer = count_layers();
		Meta map;
		map.insert(std::make_pair("Nlayers", nlayer));

		for (int i = 0; i < nlayer; ++i)
		{
			m_layers[i]->fill_meta_info(map, i);
		}

		// TODO: довести метод до ума
	}

public:
	///
	/// Стандартный конструктор
	/// 

	NeuralNetwork() :
		m_default_rng(1),
		m_rng(m_default_rng),
		m_output(NULL),
		m_default_callback(),
		m_callback(&m_default_callback)
	{}

	///
	/// Конструктор при передаче другого генератора
	/// 

	NeuralNetwork(RNG& rng) :
		m_default_rng(1),
		m_rng(rng),
		m_output(NULL),
		m_default_callback(),
		m_callback(&m_default_callback)
	{}

	///
	/// Деструктор, удаляем из памяти все слои
	/// 
	~NeuralNetwork()
	{
		const int nlayer = count_layers();

		for (int i = 0; i < nlayer; ++i)
		{
			delete m_layers[i];
		}

		if (m_output)
		{
			delete m_output;
		}
	}

	/// <summary>
	/// Подсчет кол-ва слоев
	/// </summary>
	/// <returns></returns>
	int count_layers() const
	{
		return m_layers.size();
	}

	/// <summary>
	/// Добавить слой в сетку
	/// </summary>
	/// <param name="layer"> - указатель на слой</param>
	void add_layer(Layer* layer)
	{
		m_layers.push_back(layer);
	}

	void set_output(Output* output)
	{
		if (m_output)
		{
			delete m_output;
		}

		m_output = output;
	}

	/// <summary>
	/// None
	/// </summary>
	/// <returns>Получить выходной слой</returns>
	const Output* get_output() const
	{
		return m_output;
	}

	/// <summary>
	/// Установить пользовательский вывод информации про обучение сетки.
	/// </summary>
	/// <param name="callback"> - ссылка на объект класса, который будет здесь работать.</param>
	void set_callback(Callback& callback)
	{
		m_callback = &callback;
	}

	/// <summary>
	/// Установить дефолтный вывод.
	/// </summary>
	void set_default_callback()
	{
		m_callback = &m_default_callback;
	}

};

