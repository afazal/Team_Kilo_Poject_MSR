
<h1> Note </h1>

This is a reproduction project as part of the MSR course 2022 at UniKo, CS department, SoftLang Team

<h2> Names of team/students </h2>

**Team Name**: Kilo

**Members:**

  * Ahmed Fazal (220201263 - afazal@uni-koblenz.de)
  * Usama Moin
  * Tahir Iqbal


<h2> Objective of reproduction  </h2>

The objective of this reproduction is to test whether the analysis of the contributorship mentioned in the paper can be reproduced and if similar results can be obtained while using various repositories.


<h2> Description </h2>

This paper characterizes contributor acknowledgment models in open source by analyzing thousands of projects that use a model called All Contributors. It focuses on the life cycle of projects through the model's lens and contrast its representation of contributorship. We would be reproducing the data pipeline for the list of contributors and contribution history.

<h3> Input data  </h3>

* The input data is extracted from GitHub API consisting of:

   - Forked results and fork parents 
   - Repositories list and shortlisted repositories list
   - Contributors list and size
 
<h3> Output data  </h3> 

*  We characterize contributor acknowledgment models in open source, analyze the life cycle of projects through different models.


<h2> Findings of reproduction </h2>

<h3> Process Delta  </h3>

* The project is a replication of the scraping of list of contributors from the GitHub API, running them through a data analysis process and the producing an output. The project can be replicated as it is and therefore does not require any major changes. The code files can be run in sequence along with the data files.
 
<h3> Output Delta  </h3> 

* The output files are the visualizations as well as some raw data will be generated in the /data/output folder when the input files are run.


<h2> Implementation of reproduction </h2>

<h3> Hardware requirements  </h3>

* Windows:

    - Processor: Intel i5-6600k (4 core 3.3 GHz) or AMD Ryzen 5 2400 G (4 core 3.6 GHz)
    - RAM: 8.00 GB
    - System type: 64-bit operating system, x64-based processor
    - Operating System: Windows 10

<h3> Software requirements  </h3> 

* Software
   
   - Jupyter Notebook or Spyder IDE
   - python 3.8
   - pandas
   - matplotlib

 

<h2> Validation  </h2>

To verify the output, it is best to open the generated files and go through the results in "data/output" folder.

<h2> Data  </h2>

We have used the sample dataset provided by the paper resource on figshare.com as the initial implementation. The relevant data files used can be found in the ./data/sample data folder
