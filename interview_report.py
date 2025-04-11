import os
import json
import logging
from datetime import datetime

# 配置日志
logger = logging.getLogger(__name__)

class InterviewReportGenerator:
    def __init__(self, records_dir="interview_records"):
        """初始化面试报告生成器"""
        self.records_dir = records_dir
        
        # 确保面试记录目录存在
        os.makedirs(self.records_dir, exist_ok=True)
        logger.info(f"面试报告生成器初始化完成，记录目录: {self.records_dir}")
    
    async def generate_report(self, interview_data, chat_function, candidate_info):
        """根据面试数据生成面试报告"""
        try:
            # 准备面试记录
            interview_record = {
                "interview_time": interview_data["interview_time"],
                "candidate_info": candidate_info,
                "questions_and_answers": interview_data["qa_pairs"],
                "total_questions": len(interview_data["qa_pairs"])
            }
            
            # 保存面试记录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            record_filename = os.path.join(self.records_dir, f"interview_record_{timestamp}.json")
            with open(record_filename, "w", encoding="utf-8") as f:
                json.dump(interview_record, f, ensure_ascii=False, indent=2)
            
            logger.info(f"面试记录已保存: {record_filename}")
            
            # 生成面试报告
            report_prompt = [
                {"role": "system", "content": "你是一位专业的技术面试官，负责评估候选人的技术能力。请根据面试问答内容，给出详细的评估报告。"},
                {"role": "user", "content": f"""
请根据以下面试记录，生成一份详细的面试评估报告：

候选人背景：
{candidate_info}

面试问答记录：
{json.dumps(interview_data['qa_pairs'], ensure_ascii=False, indent=2)}

请在报告中包含以下内容：
1. 候选人的技术能力评估
2. 沟通表达能力评估
3. 问题回答质量分析
4. 技术深度和广度评估
5. 总体评分（1-10分）
6. 是否推荐录用，以及理由
7. 改进建议
"""}
            ]
            
            # 调用LLM生成报告
            report_content = chat_function(report_prompt)
            
            if report_content:
                # 保存面试报告
                report_filename = os.path.join(self.records_dir, f"interview_report_{timestamp}.md")
                with open(report_filename, "w", encoding="utf-8") as f:
                    f.write(f"# 面试评估报告\n\n")
                    f.write(f"面试时间: {interview_data['interview_time']}\n\n")
                    f.write(report_content)
                
                logger.info(f"面试报告已生成: {report_filename}")
                return report_filename, report_content
            else:
                logger.error("生成面试报告失败")
                return None, "生成面试报告失败"
        except Exception as e:
            logger.error(f"生成面试报告错误: {e}")
            return None, f"生成面试报告错误: {e}"
    
    def get_all_reports(self):
        """获取所有已生成的面试报告"""
        try:
            reports = []
            if os.path.exists(self.records_dir):
                for filename in os.listdir(self.records_dir):
                    if filename.startswith("interview_report_") and filename.endswith(".md"):
                        report_path = os.path.join(self.records_dir, filename)
                        with open(report_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        
                        # 提取时间戳
                        timestamp = filename.replace("interview_report_", "").replace(".md", "")
                        
                        reports.append({
                            "filename": filename,
                            "path": report_path,
                            "timestamp": timestamp,
                            "content": content
                        })
            
            # 按时间戳排序，最新的在前
            reports.sort(key=lambda x: x["timestamp"], reverse=True)
            return reports
        except Exception as e:
            logger.error(f"获取面试报告列表错误: {e}")
            return []
    
    def get_report_by_timestamp(self, timestamp):
        """根据时间戳获取特定的面试报告"""
        try:
            report_path = os.path.join(self.records_dir, f"interview_report_{timestamp}.md")
            if os.path.exists(report_path):
                with open(report_path, "r", encoding="utf-8") as f:
                    content = f.read()
                return content
            return None
        except Exception as e:
            logger.error(f"获取面试报告错误: {e}")
            return None