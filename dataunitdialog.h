/*
 * 文件名: dataunitdialog.h
 * 文件作用: 单位与列属性管理对话框
 * 功能描述:
 * 1. 移除了顶部多余的目标单位制选择。
 * 2. 增加了预览按钮的槽函数声明。
 */

#ifndef DATAUNITDIALOG_H
#define DATAUNITDIALOG_H

#include <QDialog>
#include <QStandardItemModel>
#include <QComboBox>
#include "dataunitmanager.h"

namespace Ui {
class DataUnitDialog;
}

class DataUnitDialog : public QDialog
{
    Q_OBJECT

public:
    explicit DataUnitDialog(QStandardItemModel* dataModel, QWidget *parent = nullptr);
    ~DataUnitDialog();

    struct ConversionTask {
        int colIndex;
        QString quantityType;
        QString fromUnit;
        QString toUnit;
        QString newHeaderName;
        bool needsMathConversion;
    };

    QList<ConversionTask> getConversionTasks() const;
    bool isAppendMode() const;
    bool isSaveToFile() const;

private slots:
    void onQuantityChanged(const QString& text);
    void onBtnOkClicked();      // 确认并校验
    void onBtnPreviewClicked(); // 预览效果槽函数

private:
    void initTable();
    void applyStyle();

    Ui::DataUnitDialog *ui;
    QStandardItemModel* m_model;

    QList<QComboBox*> m_qtyCombos;
    QList<QComboBox*> m_fromUnitCombos;
    QList<QComboBox*> m_toUnitCombos;
    QStringList m_originalHeaders;
};

#endif // DATAUNITDIALOG_H
