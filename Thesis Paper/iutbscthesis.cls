\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{iutbscthesis}[2024/05/31 LaTeX Class for writing BSc Thesis Report of Islamic University of Technology (IUT), written by MBH and MIA]

% Base setup
\LoadClass[a4paper, 12pt]{report}
\RequirePackage[left=3.5cm, right=2.5cm, top=3cm, bottom=3cm]{geometry}
\RequirePackage{lmodern}
\RequirePackage{stix2}
\RequirePackage{parskip}

% Custom commands/macros to store department and program
\def\department#1{\gdef\@department{#1}}
\def\program#1{\gdef\@program{#1}}

% Line Spacing
\RequirePackage{setspace}
\onehalfspacing

% Handling floats
\RequirePackage{floatrow}

\RequirePackage{graphicx}

\RequirePackage[labelfont=bf]{caption}
\RequirePackage[labelformat=simple]{subcaption}
\renewcommand\thesubfigure{(\alph{subfigure})}

\DeclareFloatVCode{afterfigure}{\vspace{-0.5\baselineskip}}
\DeclareFloatVCode{aftertable}{\vspace{-0.25\baselineskip}}
\floatsetup[figure]{style=plain, captionskip=0.5\baselineskip, postcode=afterfigure}
\floatsetup[table]{style=plaintop, captionskip=0.5\baselineskip, postcode=aftertable}

\RequirePackage{array}
\RequirePackage{booktabs}

% Mathematics
\RequirePackage{amsmath}
%\RequirePackage{amssymb}
\RequirePackage{xfrac}

% Code listing
\RequirePackage{listings}
\RequirePackage{upquote}

% Hyperlinking
\RequirePackage[hidelinks]{hyperref}

% Bibliography
\RequirePackage[backend=biber, style=ieee, sorting=nyt]{biblatex}
\setlength{\bibitemsep}{0.8\baselineskip}

% Handling personnel (author, supervisor, and co-supervisor) (custom style file)
\RequirePackage{personnelhandler}

% Managing defense dates
\def\defenseday#1{\gdef\@defenseday{#1}}
\def\defensemonth#1{\gdef\@defensemonth{#1}}
\def\defenseyear#1{\gdef\@defenseyear{#1}}

\newcommand*{\defensedate}[3]{%
    \defenseday{#1}%
    \defensemonth{#2}%
    \defenseyear{#3}%
}

\newcommand*{\getdatemy}{%
    \@defensemonth, \@defenseyear
}

\newcommand*{\getdatemdy}{%
    \@defensemonth\ \@defenseday, \@defenseyear
}

% Change page numbering style without resetting the counter
\newcounter{savepage}
\newcommand*{\pagenumberingstyleonly}[1]{%
    \setcounter{savepage}{\value{page}}
    \pagenumbering{#1}
    \setcounter{page}{\value{savepage}}
}

% For front matter handling
\RequirePackage{frontmatter}

% Wrapper for bibliography printing
\newcommand{\printbib}{%
    \begin{spacing}{1}%
    \printbibliography[heading={bibintoc}, title={References}]%
    \end{spacing}%
}

% Appendix handling
\RequirePackage[toc,page]{appendix}