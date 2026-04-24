from utils.logger_handler import logger
from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
import random,os,requests
from utils.config_handler import agent_config
from utils.path_tool import get_abs_path

rag = RagSummarizeService()
AMAP_KEY = os.getenv("AMAP_API_KEY")

user_ids = ["1001","1002","1003","1004","1005","1006","1007","1008","1009","1010"]
month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
             "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12"]

@tool(description="从向量存储中检索参考资料")
def rag_summarize(query):
    return rag.rag_summarize(query)

@tool(description="获取指定城市的天气信息，入参为城市名称（如：杭州）")
def get_weather(city):
    """
    使用高德 API 获取实时天气
    """
    
    if not (AMAP_KEY == os.getenv("AMAP_API_KEY")):
        logger.warning("未配置高德 API Key，返回 Mock 数据")
        return f"城市{city}的天气为晴天（模拟数据）"
    
    try:
        # 1. 获取城市的 adcode
        geo_url = f"https://restapi.amap.com/v3/geocode/geo?address={city}&key={AMAP_KEY}"
        geo_resp = requests.get(geo_url, timeout=5).json()
        
        if geo_resp.get("status") == "1" and geo_resp.get("geocodes"):
            adcode = geo_resp["geocodes"][0]["adcode"]
            
            # 2. 查询天气
            weather_url = f"https://restapi.amap.com/v3/weather/weatherInfo?city={adcode}&key={AMAP_KEY}"
            weather_resp = requests.get(weather_url, timeout=5).json()
            
            if weather_resp.get("status") == "1" and weather_resp.get("lives"):
                live = weather_resp["lives"][0]
                return f"城市{city}的实时天气：{live['weather']}，气温{live['temperature']}℃，湿度{live['humidity']}%。"
        
        return f"未能获取到城市{city}的实时天气信息"
    except Exception as e:
        logger.error(f"调用高德天气接口失败: {str(e)}")
        return f"获取天气服务暂时不可用"

@tool(description="获取用户当前所在的城市名称")
def get_user_location():
    """
    使用高德 IP 定位 API 获取用户位置
    """
    if not (AMAP_KEY == os.getenv("AMAP_API_KEY")):
        logger.warning("未配置高德 API Key，返回 Mock 数据")
        return "杭州（模拟数据）"

    try:
        url = f"https://restapi.amap.com/v3/ip?key={AMAP_KEY}"
        resp = requests.get(url, timeout=5).json()
        if resp.get("status") == "1":
            city = resp.get("city")
            return city if isinstance(city, str) else "未知城市"
        return "未能识别用户所在位置"
    except Exception as e:
        logger.error(f"调用高德 IP 定位接口失败: {str(e)}")
        return "杭州"

@tool(description="获取用户id")
def get_user_id():
    return random.choice(user_ids)

@tool(description="获取当前月份")
def get_current_month():
    return random.choice(month_arr)

external_data={}
def generate_external_data():
    if not external_data:
        external_data_path = get_abs_path(agent_config["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

        with open(external_data_path,"r",encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                arr = line.strip().split(",")

                user_id = arr[0].replace('"','')
                feature = arr[1].replace('"','')
                efficiency = arr[2].replace('"','')
                consumables = arr[3].replace('"','')
                comparison = arr[4].replace('"','')
                time = arr[5].replace('"','')

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][time] = {
                    "特征" : feature,
                    "效率" : efficiency,
                    "耗材" : consumables,
                    "对比" : comparison
                }

@tool(description="从外部系统中获取指定用户指定月份的使用记录")
def fetch_external_data(user_id,month):
    generate_external_data()
    try:
        return external_data[user_id][month]
    except KeyError:
        logger.warning(f"未能检索到用户{user_id}在{month}的使用记录")
        return ""

@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"