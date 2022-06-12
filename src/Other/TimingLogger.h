#pragma once

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <QtCore/qdir.h>
#include <QtCore/qdatetime.h>
#include <QtCore/qcoreapplication.h>

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

	//Returns a string of the TimingInfo formatted as XML
	//t_depth controls the indent
	//Example:
	//<%ID% Duration="% ms" Percent="%%">
	// ...
	//</%ID%>
	std::string toXMLString(size_t t_depth = 0)
	{
		//Indent tag by depth
		std::string result(t_depth, ' ');

		//Open tag of ID with attribute for duration
		result += "<" + m_id + " Duration=\"" + std::to_string(m_duration) + " ms\"";
		//We have a parent so add an attribute for percentage of parent duration
		if (!m_parentInfo.expired())
		{
			auto parent = m_parentInfo.lock();
			result += " Percent=\"" + std::to_string((m_duration * 100) / parent->m_duration) + "%\"";
		}

		//No child info so we can close the tag straight away
		if (m_childInfo.empty())
		{
			result += "/>\n";
		}
		//Add child info as child tags
		else
		{
			result += ">\n";

			for (auto child : m_childInfo)
				result += child->toXMLString(t_depth + 1);

			//Close tag
			result += std::string(t_depth, ' ') + "</" + m_id + ">\n";
		}

		return result;
	}

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
	typedef std::pair<std::vector<std::shared_ptr<TimingInfo>>, std::shared_ptr<TimingInfoSummary>> SummaryGroup;
	typedef std::vector<std::pair<std::string, SummaryGroup>> Summary;

	TimingInfoSummary() {}
	~TimingInfoSummary() {}

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
		//Add timing info to summary
		it->second.first.push_back(t_timingInfo);

		//Add child info to child TimingInfoSummary
		if (!t_timingInfo->m_childInfo.empty())
		{
			//No child TimingInfoSummary yet, so add it
			if (it->second.second == nullptr)
				it->second.second = std::make_shared<TimingInfoSummary>();

			for (auto child : t_timingInfo->m_childInfo)
				it->second.second->add(child);
		}
	}

	//Returns a string of the TimingInfo summary as XML
	//Similar to TimingInfo::toXMLString, except it combines tags at the same level with matching ID
	//Example:
	//<%ID% Duration="% ms" Percent="%%" Count="%">
	// ...
	//<%ID%>
	std::string toXMLString(size_t t_depth = 0, double t_parentDuration = 0.0)
	{
		std::string result;

		for (auto [id, timingInfoGroup] : m_summary)
		{
			//Calculate total duration
			double duration = 0;
			for (auto timingInfo : timingInfoGroup.first)
			{
				duration += timingInfo->m_duration;
			}

			//Indent tag by depth
			result += std::string(t_depth, ' ');
			//Open tag of ID with attribute for duration
			result += "<" + id + " Duration=\"" + std::to_string(duration) + " ms\"";
			//We have a parent so add an attribute for percentage of parent duration
			if (t_parentDuration > 0.0)
			{
				result += " Percent=\"" + std::to_string((duration * 100) / t_parentDuration) + "%\"";
			}
			//Add an attribute for count
			result += " Count=\"" + std::to_string(timingInfoGroup.first.size()) + "\"";

			//No child summary so we can close the tag straight away
			if (timingInfoGroup.second == nullptr)
			{
				result += "/>\n";
			}
			//Add child summary as child tags, then close tag
			else
			{
				result += ">\n";

				result += timingInfoGroup.second->toXMLString(t_depth + 1, duration);

				//Close tag
				result += std::string(t_depth, ' ') + "</" + id + ">\n";
			}
		}

		return result;
	}

private:
	Summary m_summary;
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
		QDir folder(QCoreApplication::applicationDirPath() + "/TimingLogger/");
		if (!folder.exists())
			folder.mkpath(".");

		//Write full timing info to file
		QFile file(folder.absoluteFilePath(QDateTime::currentDateTime().toString("yyyy-MM-dd#hh-mm-ss") + ".xml"));
		file.open(QIODevice::WriteOnly | QIODevice::NewOnly);
		if (!file.isWritable())
			throw std::invalid_argument("File is not writable: " + file.fileName().toStdString());
		else
		{
			file.write(m_timingInfo->toXMLString().c_str());

			file.close();
		}

		//Write summary timing info to file
		QFile summaryFile(folder.absoluteFilePath(QDateTime::currentDateTime().toString("yyyy-MM-dd#hh-mm-ss") + "#Summary.xml"));
		summaryFile.open(QIODevice::WriteOnly | QIODevice::NewOnly);
		if (!summaryFile.isWritable())
			throw std::invalid_argument("File is not writable: " + summaryFile.fileName().toStdString());
		else
		{
			TimingInfoSummary timingInfoSummary;
			timingInfoSummary.add(m_timingInfo);
			summaryFile.write(timingInfoSummary.toXMLString().c_str());

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

private:
#ifdef TIMING_LOGGER
	std::shared_ptr<TimingInfo> m_timingInfo;
	std::weak_ptr<TimingInfo> m_latestActiveInfo;
#endif
};