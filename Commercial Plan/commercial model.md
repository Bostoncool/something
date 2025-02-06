# Job seeker and  employer

## 解决问题

1.毕业生不知道自己适合什么工作
2.毕业生距离理想岗位还有多大差距，需要完善什么技能
3.企业只知道自己的需求，不知道如何准确的撰写招聘需求
4.企业不知道给一个岗位设定多少薪资合适
5.失业人员不清楚工作经验可以去什么企业再就业
6.城市最缺少的岗位是什么
7.城市的产业结构是什么
8.敏感数据：什么样的人会去什么企业和岗位，收入是几何？


## 解决方案

1. Data collection
采用的也就是网络爬虫(Spider)技术，（Web Cralwer and Scraper），Google公司的搜索引擎（Search engine），在面临Yahoo，IBM，Microsoft等大公司的围剿之际，杀出重围，靠的就是网络爬虫技术。将互联网上的信息进行收集，并按关键词匹配，分类汇总，放到用户面前。可爬虫技术是一种典型的访问服务器获取响应的技术，你不发送请求，他绝不会响应。<font color="red">（static and passive）</font>
在抓取数据的过程中，需要遵守robot协议，而且需要及时更新，因为有的行业会没落，有的行业会兴起，有的行业会消失，有的行业会诞生，有的岗位会饱和，有的岗位会空缺。数据主要来源：APP，mini program，web，etc.


2. Data storage
正常情况下，我们会考虑到CSV，Excel等储存技术，但是这些技术在面对海量数据时，显得力不从心。我们需要一个可以存储海量数据的技术，并且可以快速查询的技术。MySQL , Redis , MongoDB , Elasticsearch , HBase , Clickhouse , 等数据库应劫而生。数据库也需要实时更新和维护，以保证数据的有效性。

3. Data preprocessing
数据会有缺失(NAN)，会有异常(abnormal)，会有重复(duplicate)，会有不一致(inconsistent)，我们需要对数据进行清洗，数据清洗是数据分析中非常重要的一步，它可以帮助我们去除数据中的噪声(noise)，提高数据的质量(quality)，从而使数据分析的结果更加准确。

4.1 Data analysis and visualization
数据分析是数据分析中非常重要的一步，基于过去的统计，我们可以预测未来的趋势。我们可以将数据可视化，将数据以图表的形式展示出来，让用户更加直观地了解数据。比如我们可以了解一个城市的产业结构，经济增长引擎，就业趋势，薪资水平，等等。我们最需要的就是了解这里的人才结构和岗位缺口，然后做好个性化匹配，数据最好实时更新。(Hadoop, Spark, Flink, Kafka, Power BI, Tableau, etc.)

4.2 Large Language Model training
这就是创新所在，这也是最为核心，最为关键的一步，我们能不能再个性化推送和岗位匹配上做到极致，就看这一步了。
将岗位的内容和求职者的内容进行匹配，然后生成一个匹配度，这个匹配度就是求职者是否适合这个岗位。不断训练迭代，不断优化反馈，不断更新模型。
国外大模型：OpenAI（ChatGPT）, Google（Gemini）, Meta（LLama）, Anthropic（Claude）, etc.
国内大模型：moonshot(Kimi) , Higt-Flyer(DeepSeek) , Alibaba(Qwen) , Bytedance(Doubao) , etc.
<font color="red">（comprehensive and accurate）</font>


1. recommendation system 
基于以下技术实现个性化推荐:

1) 协同过滤(Collaborative Filtering)
- 基于用户的协同过滤(User-CF):分析相似用户的求职偏好(求职者画像)
- 基于物品的协同过滤(Item-CF):分析相似岗位的匹配模式(岗位画像)

1) 基于内容的推荐(Content-based)
- 分析职位描述、技能要求等文本特征(Tag)
- 提取求职者简历中的关键信息(Tag)
- 计算职位与求职者画像的相似度(Tag)

1) 深度学习模型(Deep Learning Model)
- 使用深度神经网络学习职位-求职者的匹配关系
- 通过注意力机制捕捉关键特征
- 端到端训练实现精准推荐

1) 混合推荐(Hybrid Recommendation)
- 结合多种推荐算法的优势
- 动态调整各算法权重
- 综合考虑多维度特征

推荐系统在求职和招聘场景中发挥着关键作用:

1. 提高求职效率: 通过个性化推荐，求职者可以更快找到合适的职位
2. 降低招聘成本: 通过精准匹配，企业可以更有效地吸引和筛选人才
3. 增强用户体验: 提供定制化服务，提升求职者和雇主的满意度


本项目利用Python从某招聘网站抓取海量招聘数据，进行数据清洗和格式化后存储到关系型数据库中（如MySQL），利用 Django + MySQL + Echarts搭建招聘信息可视化分析系统，实现不同岗位的学历要求、工作经验、技能要求、薪资待遇等维度的可视化分析。同时依据用户需求，运用协同过滤推荐算法来实现热门岗位的推荐。

技术栈:
Django框架、协同过滤推荐算法、爬虫、MySQL、Echarts、LLM、Machine Learning.
NLP、分词器、词向量、词频、TF-IDF、余弦相似度、协同过滤、注意力机制、深度学习、混合推荐、个性化推荐、大数据、云计算





推荐系统是基于数据分析和机器学习技术，通过分析用户行为、兴趣和偏好，为用户提供个性化的推荐服务。<font color="red">（Dynamic and active）</font>



