table = readtable('spambase/spambase.dat'); 
data = table2array(table);

freq_bins = logspace(log2(0.000000001), log2(100), 7);  

% all word freq 

histogram(data(:, 1:48));

title("All 48 word frequency columns");
xlabel("Frequency");
ylabel("Frequency of this frequency");

saveas(gcf,[pwd '/img/word_freq.png']);

% all word freq outliers

dat = reshape(data(:, 1:48).',1,[]);
histogram(dat(dat > 1), 50);

title("All 48 word frequency columns (values > 1)");
xlabel("Frequency");
ylabel("Frequency of this frequency");

saveas(gcf,[pwd '/img/word_freq_outliers.png']);


% all char freq
histogram(data(:, 49:54));

title("All 6 char frequency columns");
xlabel("Frequency");
ylabel("Frequency of this frequency");

saveas(gcf,[pwd '/img/char_freq.png']);

dat = reshape(data(:, 49:54).',1,[]);
histogram(dat(dat > 1), 50);

% all char freq outliers

title("All 6 char frequency columns (values > 1)");
xlabel("Frequency");
ylabel("Frequency of this frequency");

saveas(gcf,[pwd '/img/char_freq_outliers.png']);

% all counts
histogram(data(:, 55:57), 50);

title("Capital run-length counts/avgs");
xlabel("Count");
ylabel("Frequency");

saveas(gcf,[pwd '/img/all_counts.png']);

% Outliers
dat = reshape(data(:, 55:57).',1,[]);
histogram(dat(dat > 2000), 50);

title("Capital run-length counts/avgs (values > 2000)");
xlabel("Count");
ylabel("Frequency");

saveas(gcf, [pwd '/img/outliers_counts.png']);

% Num spam
split = tabulate(data(:, end));
labels = {"Spam", "Not Spam"};
pie(split(:, 2), labels);
title("Emails in database");
saveas(gcf, [pwd '/img/spam_pie.png']);


