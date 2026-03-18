/*
 * 文件名: fittingsamplingdialog.cpp
 * 文件作用: 数据抽样设置对话框的实现
 * 功能描述:
 * 1. 实现表格的增删改查逻辑。
 * 2. 提供默认的对数空间抽样策略生成算法。
 */

#include "fittingsamplingdialog.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QHeaderView>
#include <cmath>

SamplingSettingsDialog::SamplingSettingsDialog(const QList<SamplingInterval>& intervals, bool enabled,
                                               double dataMinT, double dataMaxT, QWidget *parent)
    : QDialog(parent), m_dataMinT(dataMinT), m_dataMaxT(dataMaxT)
{
    setWindowTitle("数据抽样策略设置");
    resize(600, 450);

    QVBoxLayout* mainLayout = new QVBoxLayout(this);

    QString info = QString("当前数据时间范围: %1 ~ %2 (h)\n\n"
                           "说明: 系统将时间轴按对数空间（如0.1-1, 1-10...）划分，每个区间默认抽取10个点。\n"
                           "您可以手动调整区间范围和点数，重点关注曲线关键变化阶段（如井储、边界）。")
                       .arg(dataMinT).arg(dataMaxT);
    QLabel* lblInfo = new QLabel(info, this);
    lblInfo->setWordWrap(true);
    mainLayout->addWidget(lblInfo);

    m_chkEnable = new QCheckBox("启用自定义分段抽样 (若未勾选，则采用系统默认策略：均匀抽取200点)", this);
    m_chkEnable->setChecked(enabled);
    mainLayout->addWidget(m_chkEnable);

    m_table = new QTableWidget(this);
    m_table->setColumnCount(3);
    m_table->setHorizontalHeaderLabels(QStringList() << "起始时间(h)" << "结束时间(h)" << "抽样点数");
    m_table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    m_table->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_table->setAlternatingRowColors(true);
    mainLayout->addWidget(m_table);

    QHBoxLayout* btnLayout = new QHBoxLayout();
    QPushButton* btnAdd = new QPushButton("添加区间", this);
    QPushButton* btnDel = new QPushButton("删除选中行", this);
    QPushButton* btnReset = new QPushButton("重置为对数默认", this);

    btnLayout->addWidget(btnAdd);
    btnLayout->addWidget(btnDel);
    btnLayout->addWidget(btnReset);
    btnLayout->addStretch();
    mainLayout->addLayout(btnLayout);

    QHBoxLayout* bottomLayout = new QHBoxLayout();
    bottomLayout->addStretch();
    QPushButton* btnOk = new QPushButton("确定", this);
    QPushButton* btnCancel = new QPushButton("取消", this);
    btnOk->setDefault(true);

    bottomLayout->addWidget(btnOk);
    bottomLayout->addWidget(btnCancel);
    mainLayout->addLayout(bottomLayout);

    connect(btnAdd, &QPushButton::clicked, this, &SamplingSettingsDialog::onAddRow);
    connect(btnDel, &QPushButton::clicked, this, &SamplingSettingsDialog::onRemoveRow);
    connect(btnReset, &QPushButton::clicked, this, &SamplingSettingsDialog::onResetDefault);
    connect(btnOk, &QPushButton::clicked, this, &QDialog::accept);
    connect(btnCancel, &QPushButton::clicked, this, &QDialog::reject);

    if (intervals.isEmpty()) {
        onResetDefault();
    } else {
        for(const auto& item : intervals) {
            addRow(item.tStart, item.tEnd, item.count);
        }
    }
}

QList<SamplingInterval> SamplingSettingsDialog::getIntervals() const {
    QList<SamplingInterval> list;
    for(int i=0; i<m_table->rowCount(); ++i) {
        SamplingInterval item;
        item.tStart = m_table->item(i, 0)->text().toDouble();
        item.tEnd = m_table->item(i, 1)->text().toDouble();
        item.count = m_table->item(i, 2)->text().toInt();

        if (item.tEnd > item.tStart && item.count > 0) {
            list.append(item);
        }
    }
    return list;
}

bool SamplingSettingsDialog::isCustomSamplingEnabled() const {
    return m_chkEnable->isChecked();
}

void SamplingSettingsDialog::addRow(double start, double end, int count) {
    int row = m_table->rowCount();
    m_table->insertRow(row);
    m_table->setItem(row, 0, new QTableWidgetItem(QString::number(start)));
    m_table->setItem(row, 1, new QTableWidgetItem(QString::number(end)));
    m_table->setItem(row, 2, new QTableWidgetItem(QString::number(count)));
}

void SamplingSettingsDialog::onAddRow() {
    double start = m_dataMinT;
    double end = m_dataMaxT;
    if (m_table->rowCount() > 0) {
        bool ok;
        double lastEnd = m_table->item(m_table->rowCount()-1, 1)->text().toDouble(&ok);
        if (ok) start = lastEnd;
    }

    double safeStart = (start <= 0) ? 1e-4 : start;
    double exponent = std::floor(log10(safeStart));
    double nextPower10 = pow(10, exponent + 1);
    end = nextPower10;

    if (end > m_dataMaxT) end = m_dataMaxT;
    if (end <= start) end = start * 10.0;

    addRow(start, end, 10);
}

void SamplingSettingsDialog::onRemoveRow() {
    int row = m_table->currentRow();
    if (row >= 0) m_table->removeRow(row);
    else if (m_table->rowCount() > 0) m_table->removeRow(m_table->rowCount() - 1);
}

void SamplingSettingsDialog::onResetDefault() {
    m_table->setRowCount(0);
    double current = m_dataMinT;
    double maxVal = m_dataMaxT;

    if (current <= 1e-6) current = 1e-6;
    if (maxVal <= current) return;

    double exponent = std::floor(log10(current));
    double nextPower10 = pow(10, exponent + 1);

    while (current < maxVal) {
        double end = nextPower10;
        if (end > maxVal) end = maxVal;
        if (end > current * 1.000001) {
            addRow(current, end, 10);
        }
        current = end;
        nextPower10 *= 10.0;
        if (std::abs(current - maxVal) < 1e-9) break;
    }
}
