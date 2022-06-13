#pragma once

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <stack>
#include <QtCore/qdir.h>
#include <QtCore/qdatetime.h>
#include <QtCore/qcoreapplication.h>
#include <QtCore/qstring.h>
#include <QtCore/qtextstream.h>

//#define TIMING_LOGGER
#ifdef TIMING_LOGGER
class TimingInfo
{
public:
	//Records the start time
	TimingInfo(const std::string &t_id, std::weak_ptr<TimingInfo> t_parent) : m_id(t_id), m_startTime(std::chrono::high_resolution_clock::now()), m_duration(-1), m_parentInfo(t_parent)
	{}

	//Adds a TimingInfo as a child
	void addChild(std::shared_ptr<TimingInfo> t_child)
	{
		m_childInfo.push_back(t_child);
	}

	//Records the end time and calculates the duration
	void end()
	{
		m_endTime = std::chrono::high_resolution_clock::now();
		m_duration = std::chrono::duration_cast<std::chrono::microseconds>(m_endTime - m_startTime).count() / 1000.0;
	}

	friend class TimingLogger;
	friend class TimingInfoSummary;
	friend class TimingInfoDataStream;

private:
	const std::string m_id;

	const std::chrono::steady_clock::time_point m_startTime;
	std::chrono::steady_clock::time_point m_endTime;
	double m_duration;

	std::weak_ptr<TimingInfo> m_parentInfo;
	std::vector<std::shared_ptr<TimingInfo>> m_childInfo;
};

class TimingInfoSummary
{
public:
	//<count, duration, child summary>
	typedef std::tuple<size_t, double, std::shared_ptr<TimingInfoSummary>> SummaryGroup;
	//{<ID, summary group>, ...}
	typedef std::vector<std::pair<std::string, SummaryGroup>> Summary;

	TimingInfoSummary() {}

	void add(std::shared_ptr<TimingInfo> t_timingInfo)
	{
		Summary::iterator it = std::find_if(m_summary.begin(), m_summary.end(),
			[&t_timingInfo](std::pair<std::string, SummaryGroup> t_val)
			{
				return t_val.first == t_timingInfo->m_id;
			});

		//ID not yet in map, so add it
		if (it == m_summary.end())
		{
			m_summary.push_back(std::make_pair(t_timingInfo->m_id, SummaryGroup()));
			it = m_summary.end() - 1;
		}
		//Increment group count
		++std::get<0>(it->second);
		//Add duration to group duration
		std::get<1>(it->second) += t_timingInfo->m_duration;

		//Add child info to child TimingInfoSummary
		if (!t_timingInfo->m_childInfo.empty())
		{
			//No child TimingInfoSummary yet, so add it
			if (std::get<2>(it->second) == nullptr)
				std::get<2>(it->second) = std::make_shared<TimingInfoSummary>();

			for (auto child : t_timingInfo->m_childInfo)
				std::get<2>(it->second)->add(child);
		}
	}

	friend class TimingInfoDataStream;

private:
	Summary m_summary;
};

class TimingInfoDataStream : public QTextStream
{
public:
	explicit TimingInfoDataStream(QIODevice *t_qIODevice) : QTextStream(t_qIODevice), m_depth(0) {}

	//Returns a string of the TimingInfo formatted as XML
	//m_depth controls the indent
	//Example:
	//<%ID% Duration="% ms" Percent="%%">
	// ...
	//</%ID%>
	TimingInfoDataStream &operator<<(const TimingInfo &t_timingInfo)
	{
		QTextStream *thisParent = this;

		//Indent tag by depth
		*thisParent << QString(m_depth, ' ');

		//Open tag of ID with attribute for duration
		*thisParent << "<" << t_timingInfo.m_id.c_str() << " Duration=\"" << t_timingInfo.m_duration << " ms\"";
		//We have a parent so add an attribute for percentage of parent duration
		if (!t_timingInfo.m_parentInfo.expired())
		{
			auto parent = t_timingInfo.m_parentInfo.lock();
			*thisParent << " Percent=\"" << ((t_timingInfo.m_duration * 100) / parent->m_duration) << "%\"";
		}

		//No child info so we can close the tag straight away
		if (t_timingInfo.m_childInfo.empty())
		{
			*thisParent << "/>\n";
		}
		//Add child info as child tags
		else
		{
			*thisParent << ">\n";

			++m_depth;
			for (auto child : t_timingInfo.m_childInfo)
				*this << *child;
			--m_depth;

			//Close tag
			*thisParent << QString(m_depth, ' ') << "</" << t_timingInfo.m_id.c_str() << ">\n";
		}

		return *this;
	}

	//Returns a string of the TimingInfo summary as XML
	//Similar to TimingInfo::toXMLString, except it combines tags at the same level with matching ID
	//Example:
	//<%ID% Duration="% ms" Percent="%%" Count="%">
	// ...
	//<%ID%>
	TimingInfoDataStream &operator<<(const TimingInfoSummary &t_summary)
	{
		QTextStream *thisParent = this;

		for (auto [id, timingInfoGroup] : t_summary.m_summary)
		{
			//Calculate total duration
			const double duration = std::get<1>(timingInfoGroup);

			//Indent tag by depth
			*thisParent << QString(m_depth, ' ');
			//Open tag of ID with attribute for duration
			*thisParent << "<" << id.c_str() << " Duration=\"" << duration << " ms\"";
			//We have a parent so add an attribute for percentage of parent duration
			if (!m_durationStack.empty())
			{
				*thisParent << " Percent=\"" << ((duration * 100) / m_durationStack.top()) << "%\"";
			}
			//Add an attribute for count
			*thisParent << " Count=\"" << std::get<0>(timingInfoGroup) << "\"";

			//No child summary so we can close the tag straight away
			if (std::get<2>(timingInfoGroup) == nullptr)
			{
				*thisParent << "/>\n";
			}
			//Add child summary as child tags, then close tag
			else
			{
				*thisParent << ">\n";

				++m_depth;
				m_durationStack.push(duration);
				*this << *(std::get<2>(timingInfoGroup));
				m_durationStack.pop();
				--m_depth;

				//Close tag
				*thisParent << QString(m_depth, ' ') << "</" << id.c_str() << ">\n";
			}
		}

		return *this;
	}

private:
	//Tracks xml tag depth
	int m_depth;
	//Tracks durations of current tags (so we can get parent duration for TimingInfoSummary)
	std::stack<double> m_durationStack;
};
#endif

class TimingLogger
{
public:
	TimingLogger() {}
	~TimingLogger() {}

	void ClearTiming()
	{
#ifdef TIMING_LOGGER
		m_timingInfo.reset();
		m_latestActiveInfo.reset();
#endif
	}

	void LogTiming()
	{
#ifdef TIMING_LOGGER
		//Create folder for saving logging timing info
		QDir folder(GetApplicationDir() + "/TimingLogger/" + outputSubdir + "/");
		if (!folder.exists())
			folder.mkpath(".");

		std::string folderStr = folder.absolutePath().toStdString();
		QString dateTimeStr = QDateTime::currentDateTime().toString("yyyy-MM-dd#hh-mm-ss");

		size_t fileID = 0;
		QString fullTimingInfoFilePath = folder.absoluteFilePath(dateTimeStr + "(" + QString::number(fileID) + ").xml");
		while (QFile::exists(fullTimingInfoFilePath))
			fullTimingInfoFilePath = folder.absoluteFilePath(dateTimeStr + "(" + QString::number(++fileID) + ").xml");

		//Write full timing info to file
		QFile file(fullTimingInfoFilePath);
		file.open(QIODevice::WriteOnly | QIODevice::NewOnly);
		if (!file.isWritable())
			throw std::invalid_argument("File is not writable: " + file.fileName().toStdString());
		else
		{
			TimingInfoDataStream out(&file);
			out << *m_timingInfo;

			file.close();
		}
		
		//Write summary timing info to file
		QFile summaryFile(folder.absoluteFilePath(dateTimeStr + "(" + QString::number(fileID) + ")#Summary.xml"));
		summaryFile.open(QIODevice::WriteOnly | QIODevice::NewOnly);
		if (!summaryFile.isWritable())
			throw std::invalid_argument("File is not writable: " + summaryFile.fileName().toStdString());
		else
		{
			TimingInfoDataStream out(&summaryFile);

			TimingInfoSummary timingInfoSummary;
			timingInfoSummary.add(m_timingInfo);
			out << timingInfoSummary;

			summaryFile.close();
		}
#endif
	}

	void StartTiming(const std::string &t_id)
	{
#ifdef TIMING_LOGGER
		std::shared_ptr<TimingInfo> newInfo = std::make_shared<TimingInfo>(t_id, m_latestActiveInfo);
		std::shared_ptr<TimingInfo> activeInfo = m_latestActiveInfo.lock();
		if (activeInfo)
			activeInfo->addChild(newInfo);
		else
			m_timingInfo = newInfo;

		m_latestActiveInfo = newInfo;
#endif
	}

	void StopTiming(const std::string &t_id)
	{
#ifdef TIMING_LOGGER
		std::shared_ptr<TimingInfo> activeInfo = m_latestActiveInfo.lock();
		if (activeInfo->m_id == t_id)
		{
			activeInfo->end();
			m_latestActiveInfo = activeInfo->m_parentInfo;
		}
		else
		{
			throw std::exception("ID doesn't match latest active info!");
		}
#endif
	}

	void StopAllTiming()
	{
#ifdef TIMING_LOGGER
		std::shared_ptr<TimingInfo> activeInfo = m_latestActiveInfo.lock();
		while (activeInfo != nullptr)
		{
			activeInfo->end();
			activeInfo = activeInfo->m_parentInfo.lock();
		}
		m_latestActiveInfo.reset();
#endif
	}

	static void SetSubdir(const QString &t_subdir)
	{
		outputSubdir = t_subdir;
	}

private:
#ifdef TIMING_LOGGER
	std::shared_ptr<TimingInfo> m_timingInfo;
	std::weak_ptr<TimingInfo> m_latestActiveInfo;

	inline static QString outputSubdir;

	QString GetApplicationDir()
	{
		static QString applicationDir;
		//Only load the application dir once
		if (applicationDir.isEmpty())
		{
			//Application dir from QCoreApplication is empty, so assume we have no QCoreApplication
			if (QCoreApplication::applicationDirPath().isEmpty())
			{
				//Create a temporary QCoreApplication
				int argc = 0;
				QCoreApplication tmp(argc, 0);
				applicationDir = QCoreApplication::applicationDirPath();
			}
			else
				applicationDir = QCoreApplication::applicationDirPath();
		}
		return applicationDir;
	}
#endif
};