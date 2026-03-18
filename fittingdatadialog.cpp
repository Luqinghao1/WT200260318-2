/*
 * 文件名: fittingdatadialog.cpp
 * 文件作用: 拟合数据加载配置窗口实现文件
 * 功能描述:
 * 1. 实现界面逻辑：数据源切换、文件读取与预览。
 * 2. 实现智能列名识别，自动匹配 Time, Pressure 等列。
 * 3. 实现试井类型切换逻辑：控制初始压力(Pi)和生产时间(tp)的输入。
 * 4. 适配多文件数据源，实现项目文件切换与预览联动。
 */

#include "fittingdatadialog.h"
#include "ui_fittingdatadialog.h"

#include <QFileDialog>
#include <QMessageBox>
#include <QTextStream>
#include <QTextCodec>
#include <QDebug>
#include <QAxObject>
#include <QDir>
#include <QFileInfo>

FittingDataDialog::FittingDataDialog(const QMap<QString, QStandardItemModel*>& projectModels, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::FittingDataDialog),
    m_projectDataMap(projectModels),
    m_fileModel(new QStandardItemModel(this))
{
    ui->setupUi(this);

    ui->comboProjectFile->clear();
    for(auto it = m_projectDataMap.begin(); it != m_projectDataMap.end(); ++it) {
        QString key = it.key();
        QFileInfo fi(key);
        QString displayName = fi.fileName().isEmpty() ? key : fi.fileName();
        ui->comboProjectFile->addItem(displayName, key);
    }

    connect(ui->radioProjectData, &QRadioButton::toggled, this, &FittingDataDialog::onSourceChanged);
    connect(ui->radioExternalFile, &QRadioButton::toggled, this, &FittingDataDialog::onSourceChanged);
    connect(ui->comboProjectFile, SIGNAL(currentIndexChanged(int)), this, SLOT(onProjectFileSelectionChanged(int)));
    connect(ui->btnBrowse, &QPushButton::clicked, this, &FittingDataDialog::onBrowseFile);
    connect(ui->comboDerivative, SIGNAL(currentIndexChanged(int)), this, SLOT(onDerivColumnChanged(int)));

    // 试井类型切换
    connect(ui->radioDrawdown, &QRadioButton::toggled, this, &FittingDataDialog::onTestTypeChanged);
    connect(ui->radioBuildup, &QRadioButton::toggled, this, &FittingDataDialog::onTestTypeChanged);

    connect(ui->checkSmoothing, &QCheckBox::toggled, this, &FittingDataDialog::onSmoothingToggled);

    // 重写确定按钮
    connect(ui->buttonBox->button(QDialogButtonBox::Ok), &QPushButton::clicked, this, &FittingDataDialog::onAccepted);
    disconnect(ui->buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);

    ui->widgetFileSelect->setVisible(false);
    onTestTypeChanged();

    if (m_projectDataMap.isEmpty()) {
        ui->radioExternalFile->setChecked(true);
        ui->radioProjectData->setEnabled(false);
        ui->comboProjectFile->setEnabled(false);
    } else {
        ui->radioProjectData->setChecked(true);
        if(ui->comboProjectFile->count() > 0)
            ui->comboProjectFile->setCurrentIndex(0);
        onSourceChanged();
    }
}

FittingDataDialog::~FittingDataDialog()
{
    delete ui;
}

void FittingDataDialog::onAccepted()
{
    if (ui->comboTime->currentIndex() < 0 || ui->comboPressure->currentIndex() < 0) {
        QMessageBox::warning(this, "提示", "请选择时间列和压力列！");
        return;
    }

    if (ui->radioDrawdown->isChecked()) {
        if (ui->spinPi->value() <= 0.0001) {
            QMessageBox::warning(this, "提示", "压力降落试井需要输入有效的地层初始压力 (Pi)！");
            return;
        }
    } else {
        // 恢复试井检查 tp
        if (ui->spinTp->value() <= 0.0001) {
            QMessageBox::warning(this, "提示", "压力恢复试井需要输入有效的关井前生产时间 (tp)！");
            return;
        }
    }

    accept();
}

QStandardItemModel* FittingDataDialog::getCurrentProjectModel() const
{
    QString key = ui->comboProjectFile->currentData().toString();
    if (m_projectDataMap.contains(key)) return m_projectDataMap.value(key);
    return nullptr;
}

void FittingDataDialog::onSourceChanged()
{
    bool isProject = ui->radioProjectData->isChecked();
    ui->widgetFileSelect->setVisible(!isProject);
    ui->comboProjectFile->setEnabled(isProject);

    QStandardItemModel* targetModel = isProject ? getCurrentProjectModel() : m_fileModel;
    ui->tablePreview->clear();

    if (targetModel) {
        QStringList headers;
        for (int i = 0; i < targetModel->columnCount(); ++i) {
            headers << targetModel->headerData(i, Qt::Horizontal).toString();
        }
        ui->tablePreview->setColumnCount(headers.size());
        ui->tablePreview->setHorizontalHeaderLabels(headers);

        int rows = qMin(50, targetModel->rowCount());
        ui->tablePreview->setRowCount(rows);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < targetModel->columnCount(); ++j) {
                QStandardItem* item = targetModel->item(i, j);
                if (item) ui->tablePreview->setItem(i, j, new QTableWidgetItem(item->text()));
            }
        }
        updateColumnComboBoxes(headers);
    } else {
        ui->tablePreview->setRowCount(0);
        ui->tablePreview->setColumnCount(0);
        updateColumnComboBoxes(QStringList());
    }
}

void FittingDataDialog::onProjectFileSelectionChanged(int index)
{
    Q_UNUSED(index);
    if (ui->radioProjectData->isChecked()) onSourceChanged();
}

void FittingDataDialog::updateColumnComboBoxes(const QStringList& headers)
{
    ui->comboTime->clear();
    ui->comboPressure->clear();
    ui->comboDerivative->clear();

    ui->comboTime->addItems(headers);
    ui->comboPressure->addItems(headers);

    ui->comboDerivative->addItem("自动计算 (Bourdet)", -1);
    for(int i=0; i<headers.size(); ++i) ui->comboDerivative->addItem(headers[i], i);

    for (int i = 0; i < headers.size(); ++i) {
        QString h = headers[i].toLower();
        if (h.contains("time") || h.contains("时间") || h.contains("date")) ui->comboTime->setCurrentIndex(i);
        if (h.contains("pressure") || h.contains("压力")) ui->comboPressure->setCurrentIndex(i);
        if (h.contains("deriv") || h.contains("导数")) ui->comboDerivative->setCurrentIndex(i + 1);
    }
}

void FittingDataDialog::onTestTypeChanged()
{
    bool isDrawdown = ui->radioDrawdown->isChecked();

    // 压力降落相关控件
    ui->spinPi->setEnabled(isDrawdown);
    ui->labelPi->setEnabled(isDrawdown);
    ui->labelUnitPi->setEnabled(isDrawdown);

    // 压力恢复相关控件
    ui->spinTp->setEnabled(!isDrawdown);
    ui->labelTp->setEnabled(!isDrawdown);
    ui->labelUnitTp->setEnabled(!isDrawdown);
}

void FittingDataDialog::onBrowseFile()
{
    QString path = QFileDialog::getOpenFileName(this, "打开数据文件", "",
                                                "所有支持文件 (*.csv *.txt *.xls *.xlsx);;CSV/文本 (*.csv *.txt);;Excel (*.xls *.xlsx)");
    if (path.isEmpty()) return;

    ui->lineEditFilePath->setText(path);
    m_fileModel->clear();

    bool success = false;
    if (path.endsWith(".xls", Qt::CaseInsensitive) || path.endsWith(".xlsx", Qt::CaseInsensitive)) {
        success = parseExcelFile(path);
    } else {
        success = parseTextFile(path);
    }

    if (success) {
        onSourceChanged();
    } else {
        QMessageBox::warning(this, "错误", "文件解析失败，请检查文件格式。");
    }
}

bool FittingDataDialog::parseTextFile(const QString& filePath)
{
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) return false;

    QTextCodec* codec = QTextCodec::codecForName("UTF-8");
    QByteArray data = file.readAll();
    file.close();

    QString content = codec->toUnicode(data);
    QTextStream in(&content);
    bool headerSet = false;
    int colCount = 0;

    while (!in.atEnd()) {
        QString line = in.readLine().trimmed();
        if (line.isEmpty()) continue;

        QChar sep = ',';
        if (line.contains('\t')) sep = '\t';
        else if (line.contains(';')) sep = ';';
        else if (line.contains(' ')) sep = ' ';

        QStringList parts = line.split(sep, Qt::SkipEmptyParts);
        for(int k=0; k<parts.size(); ++k) {
            QString s = parts[k].trimmed();
            if(s.startsWith('"') && s.endsWith('"')) s = s.mid(1, s.length()-2);
            parts[k] = s;
        }

        if (!headerSet) {
            m_fileModel->setHorizontalHeaderLabels(parts);
            colCount = parts.size();
            headerSet = true;
        } else {
            QList<QStandardItem*> items;
            for (const QString& p : parts) items.append(new QStandardItem(p));
            while(items.size() < colCount) items.append(new QStandardItem(""));
            m_fileModel->appendRow(items);
        }
    }
    return true;
}

bool FittingDataDialog::parseExcelFile(const QString& filePath)
{
    QAxObject excel("Excel.Application");
    if (excel.isNull()) return false;
    excel.setProperty("Visible", false);
    excel.setProperty("DisplayAlerts", false);

    QAxObject *workbooks = excel.querySubObject("Workbooks");
    if (!workbooks) return false;
    QAxObject *workbook = workbooks->querySubObject("Open(const QString&)", QDir::toNativeSeparators(filePath));
    if (!workbook) { excel.dynamicCall("Quit()"); return false; }

    QAxObject *sheets = workbook->querySubObject("Worksheets");
    QAxObject *sheet = sheets->querySubObject("Item(int)", 1);

    if (sheet) {
        QAxObject *usedRange = sheet->querySubObject("UsedRange");
        if (usedRange) {
            QVariant varData = usedRange->dynamicCall("Value()");
            QList<QList<QVariant>> rowsData;
            if (varData.typeId() == QMetaType::QVariantList) {
                QList<QVariant> rows = varData.toList();
                for (const QVariant &row : rows) {
                    if (row.typeId() == QMetaType::QVariantList) rowsData.append(row.toList());
                }
            }
            if (!rowsData.isEmpty()) {
                QStringList headers;
                for(const QVariant& v : rowsData.first()) headers << v.toString();
                m_fileModel->setHorizontalHeaderLabels(headers);
                for(int i=1; i<rowsData.size(); ++i) {
                    QList<QStandardItem*> items;
                    for(const QVariant& v : rowsData[i]) items.append(new QStandardItem(v.toString()));
                    m_fileModel->appendRow(items);
                }
            }
            delete usedRange;
        }
        delete sheet;
    }
    workbook->dynamicCall("Close()");
    excel.dynamicCall("Quit()");
    return true;
}

void FittingDataDialog::onDerivColumnChanged(int index) { Q_UNUSED(index); }
void FittingDataDialog::onSmoothingToggled(bool checked) { ui->spinSmoothSpan->setEnabled(checked); }

FittingDataSettings FittingDataDialog::getSettings() const
{
    FittingDataSettings s;
    s.isFromProject = ui->radioProjectData->isChecked();
    if (s.isFromProject) s.projectFileName = ui->comboProjectFile->currentData().toString();
    s.filePath = ui->lineEditFilePath->text();
    s.timeColIndex = ui->comboTime->currentIndex();
    s.pressureColIndex = ui->comboPressure->currentIndex();
    s.derivColIndex = ui->comboDerivative->currentData().toInt();
    s.skipRows = ui->spinSkipRows->value();

    if (ui->radioDrawdown->isChecked()) {
        s.testType = Test_Drawdown;
        s.initialPressure = ui->spinPi->value();
        s.producingTime = 0.0;
    } else {
        s.testType = Test_Buildup;
        s.initialPressure = 0.0;
        s.producingTime = ui->spinTp->value(); // [新增]
    }

    s.lSpacing = ui->spinLSpacing->value();
    s.enableSmoothing = ui->checkSmoothing->isChecked();
    s.smoothingSpan = ui->spinSmoothSpan->value();
    return s;
}

QStandardItemModel* FittingDataDialog::getPreviewModel() const
{
    return ui->radioProjectData->isChecked() ? getCurrentProjectModel() : m_fileModel;
}
