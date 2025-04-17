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
            
            # 准备问答记录的格式化文本，用于报告
            qa_formatted = ""
            for i, qa in enumerate(interview_data["qa_pairs"]):
                # 添加问题和时间信息
                question_time = qa.get("question_time", "未记录")
                qa_formatted += f"\n问题 {i+1}: {qa['question']}\n"
                qa_formatted += f"提问时间: {question_time}\n"
                
                # 添加回答和时间信息（如果有回答）
                if qa.get("answer"):
                    answer_time = qa.get("answer_time", "未记录")
                    qa_formatted += f"面试者回答: {qa['answer']}\n"
                    qa_formatted += f"回答时间: {answer_time}\n"
                    
                    # 添加回答完成状态
                    answer_status = "已完成" if qa.get("answer_complete", False) else "未完成"
                    qa_formatted += f"回答状态: {answer_status}\n"
                    
                    # 添加用户是否跳过问题的信息
                    if qa.get("user_skipped", False):
                        qa_formatted += "注: 面试者选择跳过此问题\n"
                else:
                    qa_formatted += "面试者回答: 未回答\n"
                
                qa_formatted += "-" * 50 + "\n"
            
            # 生成面试报告
            report_prompt = [
                {"role": "system", "content": "你是一位专业的技术面试官，负责评估候选人的技术能力。请根据面试问答内容，给出详细的评估报告。"},
                {"role": "user", "content": f"""
请根据以下面试记录，生成一份详细的面试评估报告：

候选人背景：
{candidate_info}

面试时间：{interview_data["interview_time"]}

面试问答记录：
{qa_formatted}

请在报告中包含以下内容，并为每个评估点给出1-10分的具体评分：

1. 技术能力评估：
   - 技术知识深度 (1-10分)
   - 技术知识广度 (1-10分)
   - 实际项目经验 (1-10分)
   - 解决问题能力 (1-10分)

2. 沟通表达能力评估：
   - 表达清晰度 (1-10分)
   - 逻辑性 (1-10分)
   - 专业术语使用 (1-10分)

3. 问题回答质量分析：
   - 回答完整性 (1-10分)
   - 回答准确性 (1-10分)
   - 举例说明能力 (1-10分)

4. 每个问题的具体评分和分析：
   对每个问题的回答进行单独评分(1-10分)和详细分析，必须引用面试者的原始回答内容作为评分依据。

5. 总体评分计算：
   - 技术能力权重：40%
   - 沟通表达能力权重：30%
   - 问题回答质量权重：30%
   - 根据以上权重计算总分 (1-10分)

6. 录用建议和理由

请确保评分有理有据，并引用候选人的原始回答作为评分依据。对于每个评分点，请给出具体的分数，并解释为什么给出这个分数。

报告格式要求：
1. 报告开头需包含面试日期和候选人姓名
2. 每个评估部分需要有明确的标题和分数
3. 在每个问题分析部分，必须先引用原始问题和回答，再进行评分和分析
4. 总分需要精确到小数点后一位
5. 报告中必须包含完整的面试问答记录，包括问题、回答和相应的时间信息
"""}
            ]
            
            # 调用LLM生成报告
            report_content = chat_function(report_prompt)
            
            if not report_content:
                logger.error("生成面试报告失败")
                return None, None
                
            # 保存面试报告
            report_filename = os.path.join(self.records_dir, f"interview_report_{timestamp}.txt")
            with open(report_filename, "w", encoding="utf-8") as f:
                f.write(report_content)
                
            logger.info(f"面试报告已生成: {report_filename}")
            
            return report_filename, report_content
            
        except Exception as e:
            logger.error(f"生成面试报告时出错: {e}")
            logger.error(f"错误详情: {str(e)}")
            import traceback
            logger.error(f"错误堆栈: {traceback.format_exc()}")
            return None, None