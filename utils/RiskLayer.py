import torch


def cox_cost(logits, at_risk, observed,failures,ties):

    logL = 0
    # pre-calculate cumsum
    cumsum_logits = torch.cumsum(logits, dim=0)
    hazard_ratio = torch.exp(logits)
    cumsum_hazard_ratio = torch.cumsum(hazard_ratio, dim=0)
    if ties == 'noties':
        log_risk = torch.log(cumsum_hazard_ratio)

        likelihood = logits - log_risk

        uncensored_likelihood = likelihood * observed.float()

        logL = -1 * uncensored_likelihood.sum()
    else:
        # Loop for death times
            print(failures)

            for t in failures:
                tfail = failures[t]
                trisk = at_risk[t]
                d = len(tfail)
                dr = len(trisk)

                logL += -cumsum_logits[tfail[-1]] + (0 if tfail[0] == 0 else cumsum_logits[tfail[0] - 1])

                if ties == 'breslow':
                    s = cumsum_hazard_ratio[trisk[-1]]
                    logL += torch.log(s) * d
                elif ties == 'efron':
                    s = cumsum_hazard_ratio[trisk[-1]]
                    r = cumsum_hazard_ratio[tfail[-1]] - (0 if tfail[0] == 0 else cumsum_hazard_ratio[tfail[0] - 1])

                    for j in range(d):

                        logL += torch.log(s - j * r / d)

                else:
                    raise NotImplementedError('tie breaking method not recognized')

    return logL

