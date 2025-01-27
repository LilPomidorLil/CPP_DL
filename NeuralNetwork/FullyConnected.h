﻿#pragma once


# include <Eigen/Core>
# include <vector>
# include <stdexcept>
# include "Config.h"
# include "Layer.h"
# include "Random.h"


template <typename Activation>
class FullyConnected : public Layer
{
protected:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Vector::ConstAlignedMapType ConstAlignedMapVec;
    typedef Vector::AlignedMapType AlignedMapVec;
    typedef std::map<std::string, int> Meta;

    Matrix m_weight; // Веса модели
    Vector m_bias;   // Смещение весов
    Matrix m_dw;     // Производная весов
    Vector m_db;     // Производная смещения
    Matrix m_z;      // Значения нейронов до активации
    Matrix m_a;      // Значения нейронов после активации
    Matrix m_din;    // Значения нейронов после backprop

public:
    FullyConnected(const int in_size, const int out_size) :
        Layer(in_size, out_size) {}

    void init(const Scalar& mu, const Scalar& sigma, RNG& rng)
    {
        init();

        internal::set_normal_random(m_weight.data(), m_weight.size(), rng, mu, sigma);
        internal::set_normal_random(m_bias.data(), m_bias.size(), rng, mu, sigma);

        //std::cout << "Weights " << std::endl << m_weight << std::endl;
        //std::cout << "Bias " << std::endl << m_bias << std::endl;
    }

    void init()
    {
        m_weight.resize(this->m_in_size, this->m_out_size);
        m_bias.resize(this->m_out_size);
        m_dw.resize(this->m_in_size, this->m_out_size);
        m_db.resize(this->m_out_size);
    }

    /// <summary>
    /// Описание того, как толкаются данные по сетке внутри одного слоя.
    /// Сначала получаем значения нейронов просто перемножая веса и предыдущие значения нейронов.
    /// Добавляем к этому смещение.
    /// Активируем.
    /// </summary>
    /// <param name="prev_layer_data"> - матрица значений нейронов предыдущего слоя</param>
    void forward(const Matrix& prev_layer_data) 
    {
        const int ncols = prev_layer_data.cols();

        m_z.resize(this->m_out_size, ncols);
        m_z.noalias() = m_weight.transpose() * prev_layer_data;
        m_z.colwise() += m_bias;

        m_a.resize(this->m_out_size, ncols);
        Activation::activate(m_z, m_a);
    }

    /// <summary>
    /// Возврат значений нейронов после активации
    /// </summary>
    /// <returns></returns>
    const Matrix& output() { return m_a; }

    /// <summary>
    /// Получаем производные этого слоя.
    /// 
    /// Нужно получить производные по 3 вещам.
    /// 
    /// 1. Производные весов - считаем Якобиан и умножаем на предыдущий слой
    /// 2. Производные смещения - среднее по строкам производных весов
    /// 3. Производные текущих значений нейронов - текущий вес на Якобиан.
    /// 
    /// Предыдущий / следующий слой считается слева направо.
    /// </summary>
    /// <param name="prev_layer_data"> - значения нейронов предыдущего слоя</param>
    /// <param name="next_layer_data"> - значения нейронов следующего слоя</param>
    void backprop(const Matrix& prev_layer_data,
        const Matrix& next_layer_data) 
    {
        const int ncols = prev_layer_data.cols();

        Matrix& dLz = m_z;
        Activation::apply_jacobian(m_z, m_a, next_layer_data, dLz);
        m_dw.noalias() = prev_layer_data * dLz.transpose() / ncols;
        m_db.noalias() = dLz.rowwise().mean();
        m_din.resize(this->m_in_size, ncols);
        m_din.noalias() = m_weight * dLz;
    }

    /// <summary>
    /// Получить производную нейронов этого слоя
    /// </summary>
    /// <returns>ссылка на информацию</returns>
    const Matrix& backprop_data() const { return m_din; }


    /// <summary>
    /// Обновление весов и смещений используя переданный алгоритм оптимизации (см. Optimizer)
    /// </summary>
    /// <param name="opt"> - объект класса Optimizer</param>
    void update(Optimizer& opt) 
    {
        ConstAlignedMapVec dw(m_dw.data(), m_dw.size());
        ConstAlignedMapVec db(m_db.data(), m_db.size());
        AlignedMapVec      w(m_weight.data(), m_weight.size());
        AlignedMapVec      b(m_bias.data(), m_bias.size());

        opt.update(dw, w);
        opt.update(db, b);
    }

    // TODO: Доделать методы для сохранения сетки.
    std::vector<Scalar> get_parametrs() const { return std::vector<Scalar>(); }

    void set_parametrs(const std::vector<Scalar>& param) {};

    std::vector<Scalar> get_derivatives() const { return std::vector<Scalar>(); }

    std::string layer_type() const { return "FullyConnected"; }

    std::string activation_type() const { return Activation::return_type(); }

    void fill_meta_info(Meta& map, int index) const {}
};