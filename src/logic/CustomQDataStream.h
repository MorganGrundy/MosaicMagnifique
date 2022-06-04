#pragma once

#include <qdatastream.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>

class CustomQDataStream : public QDataStream
{
public:
	//
	enum class CVMatMode { RAW, ENCODE_PNG };

	explicit CustomQDataStream(QIODevice *t_qIODevice, const CVMatMode t_cvMatMode) : QDataStream(t_qIODevice), m_cvMatMode(t_cvMatMode) {}

	void SetCVMatMode(const CVMatMode t_cvMatMode)
	{
		m_cvMatMode = t_cvMatMode;
	}

	//Outputs a OpenCV mat to a QDataStream
	//Can be used to save a OpenCV mat to a file
	CustomQDataStream &operator<<(const cv::Mat &t_mat)
	{
        QDataStream *thisParent = this;

        switch (m_cvMatMode)
        {
        case CVMatMode::RAW:
        {
            *thisParent << static_cast<quint32>(t_mat.type()) << static_cast<quint32>(t_mat.rows)
                << static_cast<quint32>(t_mat.cols);

            const int dataSize = t_mat.cols * t_mat.rows * static_cast<int>(t_mat.elemSize());
            QByteArray data = QByteArray::fromRawData(reinterpret_cast<const char *>(t_mat.ptr()), dataSize);
            *thisParent << data;
        }
        break;

        case CVMatMode::ENCODE_PNG:
        {
            std::vector<uchar> buff;
            cv::imencode(".png", t_mat, buff);
            *thisParent << QByteArray(reinterpret_cast<const char *>(buff.data()), buff.size());
        }
        break;

        default:
            throw std::invalid_argument("Mode not implemented!");
        }

        return *this;
	}

	//Inputs a OpenCV mat from a QDataStream
	//Can be used to load a OpenCV mat from a file
	CustomQDataStream &operator>>(cv::Mat &t_mat)
	{
        QDataStream *thisParent = this;

        switch (m_cvMatMode)
        {
        case CVMatMode::RAW:
        {
            quint32 type, rows, cols;
            QByteArray data;
            *thisParent >> type >> rows >> cols;
            *thisParent >> data;

            t_mat = cv::Mat(rows, cols, type, data.data()).clone();
        }
        break;

        case CVMatMode::ENCODE_PNG:
        {
            QByteArray data;
            *thisParent >> data;
            std::vector<uchar> buff(data.cbegin(), data.cend());
            t_mat = cv::imdecode(buff, cv::IMREAD_UNCHANGED);
        }
        break;

        default:
            throw std::invalid_argument("Mode not implemented!");
        }

        return *this;
	}

private:
	CVMatMode m_cvMatMode;
};