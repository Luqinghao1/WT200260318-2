/*
 * 文件名: dataimportdialog.h
 * 文件作用: 数据导入配置对话框头文件
 * 功能描述:
 * 1. 定义数据导入弹窗类，用于预览文件并配置导入参数。
 * 2. 声明 Excel 预览读取功能（同时支持 QXlsx 和 QAxObject）。
 * 3. 声明防止 UI 卡顿的定时器机制。
 */

#ifndef DATAIMPORTDIALOG_H
#define DATAIMPORTDIALOG_H

#include <QDialog>
#include <QFile>
#include <QTextCodec>
#include <QTimer>
#include <QAxObject> // 保留：用于处理 .xls 文件

namespace Ui {
class DataImportDialog;
}

// 导入配置参数结构体
struct DataImportSettings {
    QString filePath;
    QString encoding;
    QString separator;
    int startRow;
    int headerRow;
    bool useHeader;
    bool isExcel; // 标记是否为 Excel 文件
};

class DataImportDialog : public QDialog
{
    Q_OBJECT

public:
    explicit DataImportDialog(const QString& filePath, QWidget *parent = nullptr);
    ~DataImportDialog();

    // 获取用户确认后的配置信息
    DataImportSettings getSettings() const;

private slots:
    // 配置改变时的槽函数（触发定时器）
    void onSettingChanged();
    // 实际执行预览更新的槽函数（由定时器触发）
    void doUpdatePreview();

private:
    Ui::DataImportDialog *ui;
    QString m_filePath;

    QList<QByteArray> m_previewLines;      // 文本文件预览缓存
    QList<QStringList> m_excelPreviewData; // Excel 文件预览缓存

    bool m_isInitializing;
    QTimer* m_previewTimer; // 防抖定时器
    bool m_isExcelFile;     // 是否检测为 Excel 文件

    // 初始化界面
    void initUI();

    // 加载文件数据（文本或 Excel）
    void loadDataForPreview();
    // 专门读取 Excel 数据的辅助函数
    void readExcelForPreview();

    // 刷新预览表格 UI
    void updatePreviewTable();

    // 获取分隔符
    QChar getSeparatorChar(const QString& sepStr, const QString& lineData);
    // 获取样式表（移除 QSpinBox border 以修复点击问题）
    QString getStyleSheet() const;
};

#endif // DATAIMPORTDIALOG_H
