/*
 * 文件名: newprojectdialog.cpp
 * 文件作用: 新建项目对话框实现文件
 * 功能描述:
 * 1. 提供新建项目向导，收集项目信息、井参数、油藏参数和PVT参数。
 * 2. 包含完整的界面美化样式表。
 * 3. [修改] 增加了水平井长度和裂缝条数的输入处理，并更新了默认参数值。
 */

#include "newprojectdialog.h"
#include "ui_newprojectdialog.h"
#include "modelparameter.h" // [新增] 引入全局参数单例
#include <QFileDialog>
#include <QMessageBox>
#include <QDir>
#include <QJsonDocument>
#include <QDebug>
#include <QDate>
#include <QStandardPaths>
#include <QStyle>

NewProjectDialog::NewProjectDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::NewProjectDialog)
{
    ui->setupUi(this);

    // 设置窗口图标 (使用系统标准文件图标)
    this->setWindowIcon(style()->standardIcon(QStyle::SP_FileIcon));

    // 加载样式 (包含日期控件颜色修复)
    loadModernStyle();

    initDefaultValues();

    // 信号连接
    connect(ui->btnBrowse, &QPushButton::clicked, this, &NewProjectDialog::on_btnBrowse_clicked);
    connect(ui->comboUnits, SIGNAL(currentIndexChanged(int)), this, SLOT(on_comboUnits_currentIndexChanged(int)));
    connect(ui->buttonBox, &QDialogButtonBox::accepted, this, &NewProjectDialog::on_btnOk_clicked);
    connect(ui->buttonBox, &QDialogButtonBox::rejected, this, &NewProjectDialog::on_btnCancel_clicked);
}

NewProjectDialog::~NewProjectDialog()
{
    delete ui;
}

void NewProjectDialog::loadModernStyle()
{
    // [完整保留原有样式表]
    QString style = R"(
        /* 全局设置 */
        QDialog {
            background-color: #ffffff;
            color: #000000;
            font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
            font-size: 10pt;
        }

        QLabel {
            color: #333333;
            font-weight: normal;
            padding: 2px;
        }

        /* 重点修复：显式包含 QDateTimeEdit
           确保文字颜色为黑色，背景为白色
        */
        QLineEdit, QDoubleSpinBox, QDateEdit, QDateTimeEdit, QComboBox {
            background-color: #ffffff;
            border: 1px solid #cccccc;
            border-radius: 4px;
            padding: 6px;
            color: #000000; /* 强制黑色文字 */
            selection-background-color: #0078d7;
            selection-color: white;
        }

        /* 聚焦状态 */
        QLineEdit:focus, QDoubleSpinBox:focus, QDateEdit:focus, QDateTimeEdit:focus, QComboBox:focus {
            border: 1px solid #0078d7;
            background-color: #fbfbfb;
        }

        /* 针对 QDateTimeEdit 内部的 QLineEdit (双重保险) */
        QDateTimeEdit QLineEdit {
            color: #000000;
            background-color: #ffffff;
        }

        QTextEdit {
            border: 1px solid #cccccc;
            border-radius: 4px;
            padding: 5px;
            background-color: white;
            color: #000000;
        }

        /* 下拉箭头样式 */
        QComboBox::drop-down, QDateEdit::drop-down, QDateTimeEdit::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 20px;
            border-left-width: 0px;
        }
        QComboBox::down-arrow, QDateEdit::down-arrow, QDateTimeEdit::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid #666;
            margin-top: 2px;
            margin-right: 2px;
        }

        /* 下拉列表视图 */
        QComboBox QAbstractItemView {
            background-color: #ffffff;
            color: #000000;
            border: 1px solid #cccccc;
            selection-background-color: #0078d7;
            selection-color: white;
        }

        /* -------------------------------------------
           日历控件样式 (QCalendarWidget)
           ------------------------------------------- */
        QCalendarWidget QWidget {
            color: #000000;
            background-color: #ffffff;
            alternate-background-color: #f9f9f9;
        }
        QCalendarWidget QWidget#qt_calendar_navigationbar {
            background-color: #ffffff;
            border-bottom: 1px solid #cccccc;
        }
        QCalendarWidget QToolButton {
            color: #000000;
            background-color: transparent;
            icon-size: 20px;
            border: none;
            font-weight: bold;
        }
        QCalendarWidget QToolButton:hover {
            background-color: #e0e0e0;
            border-radius: 4px;
        }
        QCalendarWidget QSpinBox {
            color: #000000;
            background-color: #ffffff;
            selection-background-color: #0078d7;
            selection-color: white;
        }
        QCalendarWidget QTableView {
            background-color: #ffffff;
            color: #000000;
            selection-background-color: #0078d7;
            selection-color: #ffffff;
            gridline-color: #e0e0e0;
        }
        /* ------------------------------------------- */

        QPushButton {
            background-color: #f0f0f0;
            border: 1px solid #dcdcdc;
            border-radius: 4px;
            color: #000000;
            padding: 6px 16px;
            font-weight: 500;
        }
        QPushButton:hover {
            background-color: #e0e0e0;
            border-color: #c0c0c0;
        }
        QPushButton:pressed {
            background-color: #d0d0d0;
        }

        /* Tab Widget */
        QTabWidget::pane {
            border: 1px solid #e0e0e0;
            background: #ffffff;
            border-radius: 4px;
            top: -1px;
        }
        QTabBar::tab {
            background: #f9f9f9;
            border: 1px solid #e0e0e0;
            padding: 8px 20px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            color: #555555;
        }
        QTabBar::tab:selected {
            background: #ffffff;
            border-bottom-color: #ffffff;
            color: #0078d7;
            font-weight: bold;
        }

        QGroupBox {
            font-weight: bold;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            margin-top: 12px;
            padding-top: 10px;
            color: #000000;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 0 5px;
            color: #0078d7;
        }
    )";
    this->setStyleSheet(style);
}

void NewProjectDialog::initDefaultValues()
{
    // Page 1: 基本信息
    ui->editProjectName->setText("Project_01");
    ui->editOilField->setText("ShaleOilField");
    ui->editWell->setText("Well-01");
    ui->editEngineer->setText("Admin");
    ui->dateEdit->setDateTime(QDateTime::currentDateTime());

    // 默认路径：优先 D 盘
    QString defaultPath = "D:/";
    QDir dDir(defaultPath);
    if (dDir.exists()) {
        ui->editPath->setText(defaultPath);
    } else {
        ui->editPath->setText(QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation));
    }

    // Page 2: 井与油藏参数 (使用新要求的默认值)
    ui->comboTestType->setCurrentIndex(1); // 压力恢复

    // [修改] 设置默认值
    ui->spinL->setValue(1000.0);    // 水平井长度 1000
    ui->spinNf->setValue(4.0);      // 裂缝条数 4
    ui->spinQ->setValue(10.0);      // 测试产量 10 (原50)
    ui->spinPhi->setValue(0.05);    // 孔隙度 0.05
    ui->spinH->setValue(10.0);      // 有效厚度 10 (原20)
    ui->spinRw->setValue(0.1);      // 井筒半径 0.1

    // Page 3: PVT (使用新要求的默认值)
    ui->comboUnits->setCurrentIndex((int)ProjectUnitType::Metric_SI);
    ui->spinCt->setValue(0.05);     // 综合压缩系数 0.05 (原5e-4)
    ui->spinMu->setValue(5.0);      // 粘度 5 (原0.5)
    ui->spinB->setValue(1.2);       // 体积系数 1.2 (原1.2)

    updateUnitLabels(ProjectUnitType::Metric_SI);
}

ProjectData NewProjectDialog::getProjectData() const
{
    return m_projectData;
}

void NewProjectDialog::on_btnBrowse_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(this, "选择项目存储位置",
                                                    ui->editPath->text(),
                                                    QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if (!dir.isEmpty()) {
        ui->editPath->setText(dir);
    }
}

void NewProjectDialog::on_comboUnits_currentIndexChanged(int index)
{
    ProjectUnitType newSystem = (ProjectUnitType)index;
    ProjectUnitType oldSystem = (newSystem == ProjectUnitType::Metric_SI) ? ProjectUnitType::Field_Unit : ProjectUnitType::Metric_SI;

    convertValues(oldSystem, newSystem);
    updateUnitLabels(newSystem);
}

void NewProjectDialog::updateUnitLabels(ProjectUnitType unit)
{
    if (unit == ProjectUnitType::Metric_SI) {
        ui->label_unit_L->setText("m");  // [新增]
        ui->label_unit_q->setText("m³/d");
        ui->label_unit_h->setText("m");
        ui->label_unit_rw->setText("m");
        ui->label_unit_Ct->setText("MPa⁻¹");
        ui->label_unit_mu->setText("mPa·s");
        ui->label_unit_B->setText("m³/m³");
    } else {
        ui->label_unit_L->setText("ft"); // [新增]
        ui->label_unit_q->setText("STB/d");
        ui->label_unit_h->setText("ft");
        ui->label_unit_rw->setText("ft");
        ui->label_unit_Ct->setText("psi⁻¹");
        ui->label_unit_mu->setText("cp");
        ui->label_unit_B->setText("RB/STB");
    }
}

void NewProjectDialog::convertValues(ProjectUnitType from, ProjectUnitType to)
{
    if (from == to) return;

    // 转换因子
    const double M_TO_FT = 3.28084;
    const double MPA_TO_PSI = 145.038;
    const double M3D_TO_STBD = 6.2898;

    double h = ui->spinH->value();
    double rw = ui->spinRw->value();
    double ct = ui->spinCt->value();
    double q = ui->spinQ->value();
    double L = ui->spinL->value(); // [新增]

    if (from == ProjectUnitType::Metric_SI && to == ProjectUnitType::Field_Unit) {
        // SI -> Field
        ui->spinH->setValue(h * M_TO_FT);
        ui->spinRw->setValue(rw * M_TO_FT);
        ui->spinCt->setValue(ct / MPA_TO_PSI);
        ui->spinQ->setValue(q * M3D_TO_STBD);
        ui->spinL->setValue(L * M_TO_FT); // [新增]
    }
    else if (from == ProjectUnitType::Field_Unit && to == ProjectUnitType::Metric_SI) {
        // Field -> SI
        ui->spinH->setValue(h / M_TO_FT);
        ui->spinRw->setValue(rw / M_TO_FT);
        ui->spinCt->setValue(ct * MPA_TO_PSI);
        ui->spinQ->setValue(q / M3D_TO_STBD);
        ui->spinL->setValue(L / M_TO_FT); // [新增]
    }
}

void NewProjectDialog::on_btnOk_clicked()
{
    if (ui->editProjectName->text().isEmpty() || ui->editOilField->text().isEmpty() || ui->editWell->text().isEmpty()) {
        QMessageBox::warning(this, "输入错误", "项目名称、油田名称和井名不能为空！");
        return;
    }
    if (ui->editPath->text().isEmpty()) {
        QMessageBox::warning(this, "输入错误", "请选择存储位置！");
        return;
    }

    if (createProjectStructure()) {
        QDialog::accept();
    }
}

void NewProjectDialog::on_btnCancel_clicked()
{
    QDialog::reject();
}

bool NewProjectDialog::createProjectStructure()
{
    // 文件夹命名: 油田-井名
    QString folderName = QString("%1-%2").arg(ui->editOilField->text().trimmed()).arg(ui->editWell->text().trimmed());
    QDir baseDir(ui->editPath->text());

    QString projectDirPath = baseDir.filePath(folderName);

    if (!baseDir.exists(folderName)) {
        if (!baseDir.mkpath(folderName)) {
            QMessageBox::critical(this, "错误", "无法创建项目文件夹，请检查路径权限。");
            return false;
        }
    }

    m_projectData.projectName = ui->editProjectName->text().trimmed();
    m_projectData.oilFieldName = ui->editOilField->text().trimmed();
    m_projectData.wellName = ui->editWell->text().trimmed();
    m_projectData.engineer = ui->editEngineer->text().trimmed();
    m_projectData.comments = ui->textComment->toPlainText();
    m_projectData.projectPath = projectDirPath;
    m_projectData.testType = ui->comboTestType->currentIndex();
    m_projectData.testDate = ui->dateEdit->dateTime();
    m_projectData.currentUnitSystem = (ProjectUnitType)ui->comboUnits->currentIndex();

    // [新增] 读取新参数
    m_projectData.horizLength = ui->spinL->value();
    m_projectData.fracCount = ui->spinNf->value();

    m_projectData.productionRate = ui->spinQ->value();
    m_projectData.porosity = ui->spinPhi->value();
    m_projectData.thickness = ui->spinH->value();
    m_projectData.wellRadius = ui->spinRw->value();

    m_projectData.compressibility = ui->spinCt->value();
    m_projectData.viscosity = ui->spinMu->value();
    m_projectData.volumeFactor = ui->spinB->value();

    QString fileName = m_projectData.projectName + ".pwt";
    m_projectData.fullFilePath = QDir(projectDirPath).filePath(fileName);

    // [关键] 将所有参数（包括新增的 L 和 nf）保存到全局单例
    // 注意：setParameters 的参数顺序必须与 ModelParameter::setParameters 定义一致
    ModelParameter::instance()->setParameters(
        m_projectData.porosity,
        m_projectData.thickness,
        m_projectData.viscosity,
        m_projectData.volumeFactor,
        m_projectData.compressibility,
        m_projectData.productionRate,
        m_projectData.wellRadius,
        m_projectData.horizLength, // [新增]
        m_projectData.fracCount,   // [新增]
        m_projectData.fullFilePath
        );

    saveProjectFile(m_projectData.fullFilePath, m_projectData);

    return true;
}

void NewProjectDialog::saveProjectFile(const QString &filePath, const ProjectData &data)
{
    QJsonObject root;
    root["projectName"] = data.projectName;
    root["oilField"] = data.oilFieldName;
    root["wellName"] = data.wellName;
    root["engineer"] = data.engineer;
    root["createdDate"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    root["testType"] = data.testType;

    QJsonObject reservoir;
    reservoir["unitSystem"] = (data.currentUnitSystem == ProjectUnitType::Metric_SI ? "Metric" : "Field");
    reservoir["productionRate"] = data.productionRate;
    reservoir["porosity"] = data.porosity;
    reservoir["thickness"] = data.thickness;
    reservoir["wellRadius"] = data.wellRadius;
    reservoir["horizLength"] = data.horizLength; // [新增]
    reservoir["fracCount"] = data.fracCount;     // [新增]

    QJsonObject pvt;
    pvt["compressibility"] = data.compressibility;
    pvt["viscosity"] = data.viscosity;
    pvt["volumeFactor"] = data.volumeFactor;

    root["reservoir"] = reservoir;
    root["pvt"] = pvt;

    QJsonDocument doc(root);
    QFile file(filePath);
    if (file.open(QIODevice::WriteOnly)) {
        file.write(doc.toJson());
        file.close();
    }
}
