/*
 * 文件名: fittingnewdialog.cpp
 * 文件作用: 新建分析对话框实现文件
 * 功能描述:
 * 1. 初始化对话框，填充已有分析列表。
 * 2. [修改] 使用 QTableWidget 替代原来的 QListWidget，实现对实测/理论曲线的精细化选择。
 * 3. 实现添加、删除待对比分析的交互逻辑。
 * 4. 实现输入有效性校验。
 * 5. [修复] 修正了获取复选框状态时的逻辑错误，确保能正确读取包裹在布局中的 QCheckBox 状态。
 */

#include "fittingnewdialog.h"
#include "ui_fittingnewdialog.h"
#include <QMessageBox>
#include <QDebug>
#include <QCheckBox>
#include <QHBoxLayout>
#include <QHeaderView>

FittingNewDialog::FittingNewDialog(const QStringList& existingNames, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::FittingNewDialog),
    m_existingNames(existingNames),
    m_tableSelected(nullptr)
{
    ui->setupUi(this);

    // 初始化名称，避免重复
    QString baseName = "Analysis";
    int count = 1;
    while(true) {
        QString tryName = QString("%1 %2").arg(baseName).arg(count);
        if(!existingNames.contains(tryName)) {
            ui->lineEditName->setText(tryName);
            break;
        }
        count++;
    }

    // 填充下拉框
    ui->comboSourceSingle->addItems(existingNames);
    ui->comboSourceMulti->addItems(existingNames);

    // [修改] 初始化表格控件，用于替代 listSelected
    initSelectionTable();

    // 连接信号槽
    connect(ui->comboMode, SIGNAL(currentIndexChanged(int)), this, SLOT(onModeChanged(int)));
    connect(ui->btnAdd, &QPushButton::clicked, this, &FittingNewDialog::onBtnAddClicked);

    // 连接表格的双击信号用于删除
    connect(m_tableSelected, &QTableWidget::doubleClicked, this, &FittingNewDialog::onTableDoubleClicked);

    // 拦截默认的accept，改由buttonBox触发onAccepted进行校验
    disconnect(ui->buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(ui->buttonBox, &QDialogButtonBox::accepted, this, &FittingNewDialog::onAccepted);

    // 初始化页面显示
    ui->stackedWidget->setCurrentIndex(0);
}

FittingNewDialog::~FittingNewDialog()
{
    delete ui;
}

// [新增] 初始化表格控件，插入到原有布局中
void FittingNewDialog::initSelectionTable()
{
    // 隐藏原本的 QListWidget
    ui->listSelected->setVisible(false);

    // 创建新的 QTableWidget
    m_tableSelected = new QTableWidget(this);
    m_tableSelected->setColumnCount(5);
    m_tableSelected->setHorizontalHeaderLabels({"分析名称", "实测压差", "实测导数", "理论压差", "理论导数"});

    // 设置表格样式
    m_tableSelected->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    m_tableSelected->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch); // 名称列拉伸
    m_tableSelected->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_tableSelected->setAlternatingRowColors(true);

    // 将表格添加到 listSelected 所在的布局中
    // 注意：这里假设 ui->listSelected 处于一个布局中。为了稳健，直接找到其父布局添加
    if(ui->listSelected->parentWidget() && ui->listSelected->parentWidget()->layout()) {
        ui->listSelected->parentWidget()->layout()->addWidget(m_tableSelected);
    }
}

QString FittingNewDialog::getNewName() const
{
    return ui->lineEditName->text().trimmed();
}

AnalysisCreateMode FittingNewDialog::getMode() const
{
    int idx = ui->comboMode->currentIndex();
    if (idx == 1) return AnalysisCreateMode::CopySingle;
    if (idx == 2) return AnalysisCreateMode::CopyMultiple;
    return AnalysisCreateMode::Blank;
}

QStringList FittingNewDialog::getSourceNames() const
{
    QStringList list;
    AnalysisCreateMode mode = getMode();

    if (mode == AnalysisCreateMode::CopySingle) {
        list << ui->comboSourceSingle->currentText();
    } else if (mode == AnalysisCreateMode::CopyMultiple) {
        // [修改] 从表格获取名称
        for(int i = 0; i < m_tableSelected->rowCount(); ++i) {
            list << m_tableSelected->item(i, 0)->text();
        }
    }
    return list;
}

// [修改] 获取详细的选择信息 (修复了无法读取 CheckBox 状态的 Bug)
QMap<QString, CurveSelection> FittingNewDialog::getSelectionDetails() const
{
    QMap<QString, CurveSelection> map;
    if (getMode() != AnalysisCreateMode::CopyMultiple) return map;

    for(int i = 0; i < m_tableSelected->rowCount(); ++i) {
        QString name = m_tableSelected->item(i, 0)->text();

        CurveSelection sel;
        // 获取各列的 CheckBox 状态
        // 列索引: 1:ObsP, 2:ObsD, 3:TheoP, 4:TheoD
        auto getChecked = [&](int col) -> bool {
            QWidget* widget = m_tableSelected->cellWidget(i, col);
            if (!widget) return false;

            // [修复关键点]
            // 1. 先尝试直接转换 (防止未来逻辑变为直接放入 CheckBox)
            QCheckBox* cb = qobject_cast<QCheckBox*>(widget);
            if(cb) return cb->isChecked();

            // 2. 如果转换失败，说明是一个包裹容器 (Container)，查找其子控件中的 QCheckBox
            cb = widget->findChild<QCheckBox*>();
            if(cb) return cb->isChecked();

            return false;
        };

        sel.showObsP = getChecked(1);
        sel.showObsD = getChecked(2);
        sel.showTheoP = getChecked(3);
        sel.showTheoD = getChecked(4);

        map.insert(name, sel);
    }
    return map;
}

void FittingNewDialog::onModeChanged(int index)
{
    ui->stackedWidget->setCurrentIndex(index);
}

void FittingNewDialog::onBtnAddClicked()
{
    QString current = ui->comboSourceMulti->currentText();
    if(current.isEmpty()) return;

    // 检查是否已添加
    auto items = m_tableSelected->findItems(current, Qt::MatchExactly);
    if (!items.isEmpty()) return;

    // 插入新行
    int row = m_tableSelected->rowCount();
    m_tableSelected->insertRow(row);

    // 1. 分析名称 (不可编辑)
    QTableWidgetItem* nameItem = new QTableWidgetItem(current);
    nameItem->setFlags(nameItem->flags() ^ Qt::ItemIsEditable);
    m_tableSelected->setItem(row, 0, nameItem);

    // 2. 辅助lambda：创建居中的 checkbox
    auto createCheckBox = [&](bool checked) {
        QCheckBox* cb = new QCheckBox();
        cb->setChecked(checked);
        // [注意] 为了居中，我们包裹了一层 QWidget，这导致了之前直接 cast 失败的问题
        QWidget* w = new QWidget();
        QHBoxLayout* l = new QHBoxLayout(w);
        l->setContentsMargins(0,0,0,0);
        l->setAlignment(Qt::AlignCenter);
        l->addWidget(cb);
        return w; // 返回包含 checkbox 的容器 widget
    };

    // 3. 添加四个复选框 (默认全选)
    m_tableSelected->setCellWidget(row, 1, createCheckBox(true)); // 实测压差
    m_tableSelected->setCellWidget(row, 2, createCheckBox(true)); // 实测导数
    m_tableSelected->setCellWidget(row, 3, createCheckBox(true)); // 理论压差
    m_tableSelected->setCellWidget(row, 4, createCheckBox(true)); // 理论导数
}

void FittingNewDialog::onTableDoubleClicked(const QModelIndex& index)
{
    // 双击任意单元格删除该行
    if(index.isValid()) {
        m_tableSelected->removeRow(index.row());
    }
}

void FittingNewDialog::onAccepted()
{
    if (getNewName().isEmpty()) {
        QMessageBox::warning(this, "警告", "请输入分析名称！");
        return;
    }

    AnalysisCreateMode mode = getMode();
    if (mode == AnalysisCreateMode::CopyMultiple) {
        if (m_tableSelected->rowCount() < 1) {
            QMessageBox::warning(this, "警告", "请至少添加一个需要复制的分析！");
            return;
        }
    } else if (mode == AnalysisCreateMode::CopySingle) {
        if (ui->comboSourceSingle->count() == 0) {
            QMessageBox::warning(this, "警告", "当前没有可供复制的分析！");
            return;
        }
    }

    accept();
}
