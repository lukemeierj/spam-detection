table = readtable('spambase/spambase.dat'); 
data = table2array(table);

freq_bins = logspace(log2(0.000000001), log2(100), 7);  



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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


%proportion of spam and not spam for outliers
spam = 0;
clean = 0;

for i = 1:4601
    outlier = false;
    for j = 1:48
        if data(i, 58) == 1
            spam = spam + 1;
        else
            clean = clean + 1;
        end
    end
end
labels = {"Spam", "Not Spam"};
pie([spam, clean], labels);
title("Word frequency outliers");
saveas(gcf, [pwd '/img/word_pie_1.png']);         

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

%proportion of spam and not spam for outliers
spam = 0;
clean = 0;

for i = 1:4601
    outlier = false;
    for j = 49:54
        if data(i,j) > 1
            outlier = true;
        end
    end
    if outlier == true
        if data(i, 58) == 1
            spam = spam + 1;
        else
            clean = clean + 1;
        end
    end
end
labels = {"Spam", "Not Spam"};
pie([spam, clean], labels);
title("Character frequency outliers");
saveas(gcf, [pwd '/img/char_pie_1.png']); 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

%proportion of spam and not spam for outliers
spam = 0;
clean = 0;

for i = 1:4601
    outlier = false;
    for j = 55:57
        if data(i,j) > 2000
            outlier = true;
        end
    end
    if outlier == true
        if data(i, 58) == 1
            spam = spam + 1;
        else
            clean = clean + 1;
        end
    end
end
labels = {"Spam", "Not Spam"};
pie([spam, clean], labels);
title("Capital run-length outliers");
saveas(gcf, [pwd '/img/caps_pie_2000.png']); 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Num spam
split = tabulate(data(:, end));
labels = {"Spam", "Not Spam"};
pie(split(:, 2), labels);
title("Emails in database");
saveas(gcf, [pwd '/img/spam_pie.png']);


