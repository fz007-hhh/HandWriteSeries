import json


def isTimeColumn(col: str) -> bool:
    """判断列是否为时间列"""
    col = col.lower()
    if (
        "time" in col
        or "date" in col
        or "week" in col
        or "month" in col
        or "year" in col
        or "时间" in col
        or "日期" in col
        or "周次" in col
        or "月份" in col
        or "年份" in col
    ):
        return True
    return False


BASE_ECHART_OPTION = {
    "title": {"text": ""},
    "tooltip": {},
    "width": 800,
    "grid": {
        "left": "5%",
        "right": "5%",
        "top": "15%",
        "bottom": "15%",
        "containLabel": True,
    },
    "xAxis": {
        "type": "category",
    },
    "yAxis": {
        "type": "value",
    },
    "responsive": True,
    "media": [
        {
            "query": {"maxWidth": 300},
            "option": {
                "grid": {
                    "left": "5%",
                    "right": "5%",
                    "bottom": "10%",
                    "top": "15%",
                },
                "title": {"fontSize": 12},
                "xAxis": {"axisLabel": {"fontSize": 10}},
                "yAxis": {"axisLabel": {"fontSize": 10}},
            },
        }
    ],
}


def build_pie_chart(dataset, itemName, value) -> dict[str, any]:
    """构建饼图配置"""
    radius = "50%"
    return {
        **BASE_ECHART_OPTION,
        "dataset": {"source": dataset},
        "series": [
            {
                "type": "pie",
                "radius": radius,
                "encode": {
                    "itemName": itemName,
                    "value": value,
                },
            }
        ],
    }


def build_line_chart(dataset, xAxis, yAxis) -> dict[str, any]:
    """构建折线图配置"""
    return {
        **BASE_ECHART_OPTION,
        "dataset": {"source": dataset},
        "xAxis": {"type": "category", "name": xAxis, "boundaryGap": False},
        "yAxis": {"type": "value"},
        "series": [{"type": "line", "encode": {"x": xAxis, "y": y}} for y in yAxis],
    }


def build_bar_chart(dataset, xAxis, yAxis) -> dict[str, any]:
    """构建柱状图配置"""
    return {
        **BASE_ECHART_OPTION,
        "dataset": {"source": dataset},
        "xAxis": {"type": "category", "name": xAxis, "scale": True},
        "yAxis": {"type": "value"},
        "series": [{"type": "bar", "encode": {"x": xAxis, "y": y}} for y in yAxis],
    }


def build_scatter_chart(dataset, xAxis, yAxis) -> dict[str, any]:
    """构建散点图配置"""
    return {
        **BASE_ECHART_OPTION,
        "dataset": {"source": dataset},
        "xAxis": {"type": "value", "name": xAxis},
        "yAxis": {"type": "value"},
        "series": [{"type": "scatter", "encode": {"x": xAxis, "y": yAxis}}],
    }


def data_pivot(data: list[dict], index_col: str, column_col: str, value_col: str):
    # 初始化一个字典来存储结果
    pivot_data = {}

    # 遍历数据并填充透视表
    for row in data:
        index = row[index_col]
        column = row[column_col]
        value = row[value_col]

        # 如果index_val不在字典中，初始化
        if index not in pivot_data:
            pivot_data[index] = {}

        # 添加数据
        pivot_data[index][column] = value

    # 转换为二维表格格式
    column_header_list = sorted(
        set(str(column) for row in data for column in [row[column_col]])
    )

    # 构建表头
    table = [[index_col] + column_header_list]

    # 填充表格内容
    for index, items in pivot_data.items():
        row = [index] + [items.get(col, None) for col in column_header_list]
        table.append(row)

    return table


def handle_data(data: list[dict]) -> list[list[any]]:
    """处理数据，将数据转换为列表形式"""

    # 1. 识别维度列
    headers = list(data[0].keys())
    # data = data[1:]

    # 动态分析数据结构
    date_col = ""
    numeric_cols = []
    categorical_cols = []
    for col in headers:
        unique_values = set([item[col] for item in data])
        if not isTimeColumn(col) and all(
            isinstance(item[col], (int, float)) for item in data
        ):
            numeric_cols.append(col)
        elif len(unique_values) > 1:
            if isTimeColumn(col):
                date_col = col
            categorical_cols.append(col)

    if len(categorical_cols) == 0 or len(categorical_cols) > 2:
        return []

    if len(numeric_cols) == 0:
        return []

    if len(numeric_cols) > 1 and len(categorical_cols) > 1:
        # 当前解决不了
        return []

    # 浮点数据保留两个小数
    for row in data:
        for item in numeric_cols:
            value = row.get(item)
            if isinstance(value, float):
                row[item] = round(value, 2)

    dataset = []
    if len(categorical_cols) == 1:
        dataset = [headers] + [[item.get(col) for col in headers] for item in data]
    elif len(categorical_cols) == 2:
        if not date_col:
            dataset = data_pivot(
                data, categorical_cols[0], categorical_cols[1], numeric_cols[0]
            )
        else:
            column_col = (
                categorical_cols[0]
                if categorical_cols[0] != date_col
                else categorical_cols[1]
            )
            dataset = data_pivot(data, date_col, column_col, numeric_cols[0])

    return dataset


def main(json_string: str, chart_type: str) -> dict:
    """如果是趋势图，则提取时间列和数值列
    如果是柱状图，则提取分类列和数值列
    如果是饼图，则提取分类列和占比列
    如果是散点图，则提取x轴和y轴列
    """
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError:
        return {
            "code": -1,
            "data": "No Echarts",
            "message": "Error: Invalid JSON data.",
        }

    if not data:
        return {
            "code": -1,
            "data": "No Echarts",
            "message": "Error: JSON data is empty.",
        }

    if len(data) <= 2:
        return {
            "code": -1,
            "data": "No Echarts",
            "message": "Error: JSON data is too small.",
        }

    dataset = handle_data(data)
    # 为安全起见，如果dataset为空，那么直接返回""
    if len(dataset) == 0:
        return {"code": 0, "data": "No Echarts", "message": ""}

    headers = dataset[0]

    # 根据图表类型构建配置
    config = {}
    if chart_type == "line":
        config = build_line_chart(dataset, headers[0], headers[1:])
    elif chart_type == "pie":
        config = build_pie_chart(dataset, headers[0], headers[1])
    elif chart_type == "scatter":
        config = build_scatter_chart(dataset, headers[0], headers[1:])
    else:
        config = build_bar_chart(dataset, headers[0], headers[1:])

    print(json.dumps(config, indent=2, ensure_ascii=False))
    output = "```echarts\n" + json.dumps(config, indent=2, ensure_ascii=False) + "\n```"
    return {"code": 0, "data": output, "message": ""}
