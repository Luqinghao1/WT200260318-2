/*
 * 文件名: modelselect.cpp
 * 文件作用: 模型选择对话框逻辑实现
 * 功能描述:
 * 1. 初始化模型选择 UI，增加了 Fair 和 Hegeman 井储模型选项。
 * 2. 实现了 5 个维度（井、储层、边界、井储、介质组合）到 72 个模型 ID 的双向映射。
 * 3. 严格遵循 modelsolver1.csv 和 modelsolver2.csv 的模型定义顺序。
 */

#include "modelselect.h"
#include "ui_modelselect.h"
#include <QDialogButtonBox>
#include <QPushButton>
#include <QDebug>

ModelSelect::ModelSelect(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ModelSelect),
    m_isInitializing(false)
{
    ui->setupUi(this);
    this->setStyleSheet("QWidget { color: black; font-family: Arial; }");

    // 初始化下拉框选项
    initOptions();

    // 信号槽连接
    connect(ui->comboReservoirModel, SIGNAL(currentIndexChanged(int)), this, SLOT(updateInnerOuterOptions()));
    connect(ui->comboWellModel, SIGNAL(currentIndexChanged(int)), this, SLOT(onSelectionChanged()));
    connect(ui->comboReservoirModel, SIGNAL(currentIndexChanged(int)), this, SLOT(onSelectionChanged()));
    connect(ui->comboBoundary, SIGNAL(currentIndexChanged(int)), this, SLOT(onSelectionChanged()));
    connect(ui->comboStorage, SIGNAL(currentIndexChanged(int)), this, SLOT(onSelectionChanged()));
    connect(ui->comboInnerOuter, SIGNAL(currentIndexChanged(int)), this, SLOT(onSelectionChanged()));

    disconnect(ui->buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(ui->buttonBox, &QDialogButtonBox::accepted, this, &ModelSelect::onAccepted);
    connect(ui->buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

    // 初始触发一次选择更新
    onSelectionChanged();
}

ModelSelect::~ModelSelect()
{
    delete ui;
}

void ModelSelect::initOptions()
{
    m_isInitializing = true;

    ui->comboWellModel->clear();
    ui->comboReservoirModel->clear();
    ui->comboBoundary->clear();
    ui->comboStorage->clear();
    ui->comboInnerOuter->clear();

    // 1. 井模型
    ui->comboWellModel->addItem("压裂水平井", "FracHorizontal");

    // 2. 储层模型 (涵盖 72 个模型的三大类)
    // Solver 1: 径向复合(25-36), 夹层型(1-24)
    // Solver 2: 页岩型(37-72)
    ui->comboReservoirModel->addItem("径向复合模型", "RadialComposite");
    ui->comboReservoirModel->addItem("夹层型径向复合模型", "InterlayerComposite");
    ui->comboReservoirModel->addItem("页岩型径向复合模型", "ShaleComposite");
    ui->comboReservoirModel->addItem("混积型径向复合模型", "MixedComposite"); // 预留

    // 3. 边界条件
    ui->comboBoundary->addItem("无限大外边界", "Infinite");
    ui->comboBoundary->addItem("封闭边界", "Closed");
    ui->comboBoundary->addItem("定压边界", "ConstantPressure");

    // 4. 井储模型 [修改：增加 Fair 和 Hegeman]
    ui->comboStorage->addItem("定井储", "Constant");
    ui->comboStorage->addItem("线源解", "LineSource");
    ui->comboStorage->addItem("Fair模型", "Fair");
    ui->comboStorage->addItem("Hegeman模型", "Hegeman");

    ui->comboWellModel->setCurrentIndex(0);
    ui->comboReservoirModel->setCurrentIndex(0);
    ui->comboBoundary->setCurrentIndex(0);
    ui->comboStorage->setCurrentIndex(0);

    m_isInitializing = false;
    updateInnerOuterOptions();
}

void ModelSelect::updateInnerOuterOptions()
{
    bool oldState = ui->comboInnerOuter->blockSignals(true);
    ui->comboInnerOuter->clear();
    QString currentRes = ui->comboReservoirModel->currentData().toString();

    // 根据储层模型动态填充内外区组合
    if (currentRes == "RadialComposite") {
        ui->comboInnerOuter->addItem("均质+均质", "Homo_Homo");
    }
    else if (currentRes == "InterlayerComposite") {
        ui->comboInnerOuter->addItem("夹层型+夹层型", "Interlayer_Interlayer");
        ui->comboInnerOuter->addItem("夹层型+均质", "Interlayer_Homo");
    }
    else if (currentRes == "ShaleComposite") {
        ui->comboInnerOuter->addItem("页岩型+页岩型", "Shale_Shale");
        ui->comboInnerOuter->addItem("页岩型+均质", "Shale_Homo");
        ui->comboInnerOuter->addItem("页岩型+双重孔隙", "Shale_Dual");
    }
    else if (currentRes == "MixedComposite") {
        ui->comboInnerOuter->addItem("混积型+混积型", "Mixed_Mixed");
        ui->comboInnerOuter->addItem("混积型+均质", "Mixed_Homo");
        ui->comboInnerOuter->addItem("混积型+双重孔隙", "Mixed_Dual");
    }

    if (ui->comboInnerOuter->count() > 0) {
        ui->comboInnerOuter->setCurrentIndex(0);
    }

    ui->comboInnerOuter->blockSignals(oldState);
    ui->label_InnerOuter->setVisible(true);
    ui->comboInnerOuter->setVisible(true);
}

// 根据模型代码 ID (1-72) 反向设置 UI 选项
void ModelSelect::setCurrentModelCode(const QString& code)
{
    m_isInitializing = true;
    QString numStr = code;
    numStr.remove("modelwidget");
    int id = numStr.toInt(); // 1-72

    if (id >= 1) {
        int idxWell = ui->comboWellModel->findData("FracHorizontal");
        if (idxWell >= 0) ui->comboWellModel->setCurrentIndex(idxWell);

        QString resData, ioData;

        // --- 储层与介质映射 ---
        if (id >= 1 && id <= 12) {
            resData = "InterlayerComposite"; ioData = "Interlayer_Interlayer";
        } else if (id >= 13 && id <= 24) {
            resData = "InterlayerComposite"; ioData = "Interlayer_Homo";
        } else if (id >= 25 && id <= 36) {
            resData = "RadialComposite"; ioData = "Homo_Homo";
        } else if (id >= 37 && id <= 48) {
            resData = "ShaleComposite"; ioData = "Shale_Shale";
        } else if (id >= 49 && id <= 60) {
            resData = "ShaleComposite"; ioData = "Shale_Homo";
        } else if (id >= 61 && id <= 72) {
            resData = "ShaleComposite"; ioData = "Shale_Dual";
        }

        int idxRes = ui->comboReservoirModel->findData(resData);
        if (idxRes >= 0) {
            ui->comboReservoirModel->setCurrentIndex(idxRes);
            updateInnerOuterOptions();
        }

        // --- 边界映射 ---
        // 规律: 12个一组，前4个Infinite，中4个Closed，后4个ConstantP
        int groupOffset = (id - 1) % 12;
        QString bndData;
        if (groupOffset < 4) bndData = "Infinite";
        else if (groupOffset < 8) bndData = "Closed";
        else bndData = "ConstantPressure";
        int idxBnd = ui->comboBoundary->findData(bndData);
        if (idxBnd >= 0) ui->comboBoundary->setCurrentIndex(idxBnd);

        // --- 井储映射 ---
        // 规律: 4个循环
        int storageOffset = (id - 1) % 4;
        QString storeData;
        if (storageOffset == 0) storeData = "Constant";
        else if (storageOffset == 1) storeData = "LineSource";
        else if (storageOffset == 2) storeData = "Fair";
        else storeData = "Hegeman";

        int idxStore = ui->comboStorage->findData(storeData);
        if (idxStore >= 0) ui->comboStorage->setCurrentIndex(idxStore);

        // 设置介质组合
        int idxIo = ui->comboInnerOuter->findData(ioData);
        if (idxIo >= 0) ui->comboInnerOuter->setCurrentIndex(idxIo);
    }

    m_isInitializing = false;
    onSelectionChanged();
}

// UI 选项改变时，计算模型 ID
void ModelSelect::onSelectionChanged()
{
    if (m_isInitializing) return;

    QString well = ui->comboWellModel->currentData().toString();
    QString res = ui->comboReservoirModel->currentData().toString();
    QString bnd = ui->comboBoundary->currentData().toString();
    QString store = ui->comboStorage->currentData().toString();
    QString io = ui->comboInnerOuter->currentData().toString();

    m_selectedModelCode = "";
    m_selectedModelName = "";

    // 辅助 lambda：根据基础 ID 和选项偏移计算最终 ID
    // 基础 ID 是该大类（介质组合）的起始 ID
    auto calcID = [&](int startId, QString bndType, QString storeType) -> int {
        int offsetBnd = 0;
        if (bndType == "Closed") offsetBnd = 4;
        else if (bndType == "ConstantPressure") offsetBnd = 8;

        int offsetStore = 0;
        if (storeType == "LineSource") offsetStore = 1;
        else if (storeType == "Fair") offsetStore = 2;
        else if (storeType == "Hegeman") offsetStore = 3;

        return startId + offsetBnd + offsetStore;
    };

    int baseStartId = 0;
    QString baseNameCn = "";

    // 确定大类起始 ID
    if (well == "FracHorizontal") {
        if (res == "InterlayerComposite") {
            if (io == "Interlayer_Interlayer") { baseStartId = 1; baseNameCn = "夹层型储层试井解释模型"; }
            else if (io == "Interlayer_Homo") { baseStartId = 13; baseNameCn = "夹层型储层试井解释模型"; }
        } else if (res == "RadialComposite") {
            if (io == "Homo_Homo") { baseStartId = 25; baseNameCn = "径向复合模型"; }
        } else if (res == "ShaleComposite") {
            if (io == "Shale_Shale") { baseStartId = 37; baseNameCn = "页岩型储层试井解释模型"; }
            else if (io == "Shale_Homo") { baseStartId = 49; baseNameCn = "页岩型储层试井解释模型"; }
            else if (io == "Shale_Dual") { baseStartId = 61; baseNameCn = "页岩型储层试井解释模型"; }
        }
    }

    if (baseStartId > 0) {
        int finalId = calcID(baseStartId, bnd, store);
        m_selectedModelCode = QString("modelwidget%1").arg(finalId);

        // 显示名称调整 (保持与 Excel 表格一致的编号显示)
        int displayId = 0;
        // 径向复合模型 ID 是 1-12 (对应全局 25-36)
        // 页岩型 ID 是 1-36 (对应全局 37-72)
        // 夹层型 ID 是 1-24 (对应全局 1-24)
        if (baseStartId == 25) displayId = finalId - 24;
        else if (baseStartId >= 37) displayId = finalId - 36;
        else displayId = finalId;

        m_selectedModelName = QString("%1%2").arg(baseNameCn).arg(displayId);
    }

    bool isValid = !m_selectedModelCode.isEmpty();
    if (isValid) {
        ui->label_ModelName->setText(m_selectedModelName);
        ui->label_ModelName->setStyleSheet("color: black; font-weight: bold; font-size: 14px;");
    } else {
        ui->label_ModelName->setText("该组合暂无已实现模型");
        ui->label_ModelName->setStyleSheet("color: red; font-weight: normal;");
    }

    QPushButton* okBtn = ui->buttonBox->button(QDialogButtonBox::Ok);
    if(okBtn) okBtn->setEnabled(isValid);
}

void ModelSelect::onAccepted() {
    if (!m_selectedModelCode.isEmpty()) accept();
}

QString ModelSelect::getSelectedModelCode() const { return m_selectedModelCode; }
QString ModelSelect::getSelectedModelName() const { return m_selectedModelName; }
