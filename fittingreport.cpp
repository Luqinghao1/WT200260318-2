/*
 * 文件名: fittingreport.cpp
 * 文件作用: 试井拟合报告生成器实现文件
 * 功能描述:
 * 1. 实现了基于 HTML 模板的报告生成逻辑。
 * 2. 自动在报告同级目录下生成数据 CSV 文件。
 * 3. 封装了文件 I/O 操作和编码处理。
 */

#include "fittingreport.h"
#include <QFile>
#include <QTextStream>
#include <QFileInfo>
#include <QDateTime>
#include <QDir>

bool FittingReportGenerator::generate(const QString& filePath, const FittingReportData& data, QString* errorMsg)
{
    if (filePath.isEmpty()) {
        if (errorMsg) *errorMsg = "文件路径为空";
        return false;
    }

    QFileInfo fileInfo(filePath);
    QString baseName = fileInfo.completeBaseName();
    QString path = fileInfo.path();

    // 1. 导出关联的数据表 CSV
    QString dataFileName = baseName + "_数据表.csv";
    QString dataFilePath = path + "/" + dataFileName;

    if (!generateDataCSV(dataFilePath, data)) {
        if (errorMsg) *errorMsg = "无法保存关联数据表文件: " + dataFilePath;
        return false;
    }

    // 2. 生成 HTML 内容
    QString htmlContent = buildHtmlContent(data, dataFileName);

    // 3. 写入报告文件
    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        if (errorMsg) *errorMsg = "无法打开报告文件进行写入: " + file.errorString();
        return false;
    }

    QTextStream out(&file);
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    out.setEncoding(QStringConverter::Utf8);
#else
    out.setCodec("UTF-8");
#endif
    // 写入 BOM 以防止中文乱码 (主要针对 Windows)
    file.write("\xEF\xBB\xBF");
    out << htmlContent;
    file.close();

    return true;
}

bool FittingReportGenerator::generateDataCSV(const QString& csvPath, const FittingReportData& data)
{
    QFile dataFile(csvPath);
    if (!dataFile.open(QIODevice::WriteOnly | QIODevice::Text)) return false;

    QTextStream outData(&dataFile);
    dataFile.write("\xEF\xBB\xBF"); // BOM
    outData << QString("序号,时间(h),压差(MPa),压力导数(MPa)\n");
    for(int i = 0; i < data.t.size(); ++i) {
        double dVal = (i < data.d.size()) ? data.d[i] : 0.0;
        outData << QString("%1,%2,%3,%4\n").arg(i+1)
                       .arg(data.t[i])
                       .arg(data.p[i])
                       .arg(dVal);
    }
    dataFile.close();
    return true;
}

QString FittingReportGenerator::buildHtmlContent(const FittingReportData& data, const QString& csvFileName)
{
    QString html = "<html xmlns:o='urn:schemas-microsoft-com:office:office' xmlns:w='urn:schemas-microsoft-com:office:word' xmlns='http://www.w3.org/TR/REC-html40'>";
    html += "<head><meta charset='utf-8'><title>Report</title><style>";
    html += "body { font-family: 'Times New Roman', 'SimSun'; font-size: 10.5pt; }";
    html += "h1 { text-align: center; font-size: 16pt; font-weight: bold; margin: 20px 0; font-family: 'SimSun'; }";
    html += "h2 { font-size: 14pt; font-weight: bold; margin-top: 15px; font-family: 'SimSun'; }";
    html += "p { margin: 3px 0; line-height: 1.5; }";
    html += "table { border-collapse: collapse; width: 100%; margin: 5px 0; font-size: 10.5pt; }";
    html += "th, td { border: 1px solid black; padding: 2px 4px; text-align: center; }";
    html += "th { background-color: #f2f2f2; font-family: 'SimSun'; }";
    html += ".img-box { text-align: center; margin: 10px 0; }";
    html += ".img-cap { font-size: 9pt; font-weight: bold; margin-top: 2px; font-family: 'SimSun'; }";
    html += ".page-break { page-break-before: always; }";
    html += "</style></head><body>";

    // 标题
    QString title = data.wellName.isEmpty() ? "试井解释报告" : (data.wellName + "试井解释报告");
    html += QString("<h1>%1</h1>").arg(title);

    // 基本信息
    QString dateStr = QDateTime::currentDateTime().toString("yyyy-MM-dd");
    QString modelStr = ModelManager::getModelTypeName(data.modelType);

    html += QString("<p><b>井名：</b>%1</p>").arg(data.wellName);
    html += QString("<p><b>报告日期：</b>%1</p>").arg(dateStr);
    html += QString("<p><b>解释模型：</b>%1</p>").arg(modelStr);
    html += QString("<p><b>数据文件：</b>%1</p>").arg(csvFileName);
    html += QString("<p><b>拟合精度 (MSE)：</b>%1</p>").arg(data.mse, 0, 'e', 4);

    // 一、数据信息摘要
    html += "<h2>一、数据信息</h2><table><tr><th>序号</th><th>时间 (h)</th><th>压差 (MPa)</th><th>压力导数 (MPa)</th></tr>";
    int rowCount = qMin(20, (int)data.t.size()); // 报告中只展示前20行，避免太长
    for(int i=0; i<rowCount; ++i) {
        double dVal = (i < data.d.size()) ? data.d[i] : 0.0;
        QString dStr = (i < data.d.size()) ? QString::number(dVal, 'f', 4) : "-";
        html += QString("<tr><td>%1</td><td>%2</td><td>%3</td><td>%4</td></tr>")
                    .arg(i+1)
                    .arg(QString::number(data.t[i], 'f', 4))
                    .arg(QString::number(data.p[i], 'f', 4))
                    .arg(dStr);
    }
    html += "</table>";
    html += QString("<p style='font-size:9pt; color:blue; text-align:right;'>* 注：以上展示前%1行数据，完整数据见附件：<b>%2</b></p>")
                .arg(rowCount).arg(csvFileName);

    // 二、拟合曲线
    html += "<br class='page-break' /><h2>二、拟合曲线</h2>";
    if (!data.imgLogLog.isEmpty()) {
        html += "<div class='img-box'>" + QString("<img src='data:image/png;base64,%1' width='500' /><br/>").arg(data.imgLogLog) + "<div class='img-cap'>图1 双对数拟合结果图</div></div>";
    }
    if (!data.imgSemiLog.isEmpty()) {
        html += "<div class='img-box'>" + QString("<img src='data:image/png;base64,%1' width='500' /><br/>").arg(data.imgSemiLog) + "<div class='img-cap'>图2 半对数坐标系压力历史图</div></div>";
    }
    if (!data.imgCartesian.isEmpty()) {
        html += "<div class='img-box'>" + QString("<img src='data:image/png;base64,%1' width='500' /><br/>").arg(data.imgCartesian) + "<div class='img-cap'>图3 标准坐标系压力历史图 (笛卡尔)</div></div>";
    }

    // 三、四、参数表
    QString fitParamRows;
    QString defaultParamRows;
    int idxFit = 1, idxDef = 1;

    for(const auto& p : data.params) {
        QString chName, symbol, uniSym, unit;
        FittingParameterChart::getParamDisplayInfo(p.name, chName, symbol, uniSym, unit);
        if(unit == "无因次" || unit == "小数") unit = "-";

        QString row = QString("<tr><td>%1</td><td>%2</td><td>%3</td><td>%4</td><td>%5</td></tr>")
                          .arg(p.isFit ? idxFit++ : idxDef++)
                          .arg(chName)
                          .arg(uniSym)
                          .arg(p.value, 0, 'g', 6)
                          .arg(unit);

        if (p.isFit) fitParamRows += row; else defaultParamRows += row;
    }

    html += "<h2>三、拟合参数</h2>";
    if (!fitParamRows.isEmpty()) {
        html += "<table><tr><th width='10%'>序号</th><th width='30%'>参数名称</th><th width='20%'>符号</th><th width='25%'>数值</th><th width='15%'>单位</th></tr>" + fitParamRows + "</table>";
    } else {
        html += "<p>无拟合参数。</p>";
    }

    html += "<h2>四、默认参数</h2>";
    if (!defaultParamRows.isEmpty()) {
        html += "<table><tr><th width='10%'>序号</th><th width='30%'>参数名称</th><th width='20%'>符号</th><th width='25%'>数值</th><th width='15%'>单位</th></tr>" + defaultParamRows + "</table>";
    } else {
        html += "<p>无默认参数。</p>";
    }

    html += "<br/><hr/><p style='text-align:center; font-size:9pt; color:#888;'>报告来自PWT页岩油多级压裂水平井试井解释软件</p></body></html>";
    return html;
}
