def get_checkpoint_resume(hparams):
    resume_from_checkpoint = hparams.resume_from_checkpoint
    if resume_from_checkpoint is None:
        return None
    if not hparams.debug:
        if hparams.fold_i in resume_from_checkpoint:
            return resume_from_checkpoint[hparams.fold_i]
        else:
            return None
    return None


def get_the_lastest_fold(hparams):
    return max(hparams.resume_from_checkpoint.keys())


def is_skip_current_fold(current_fold, hparams):
    if hparams.resume_from_checkpoint is None:
        return False
    folds = hparams.skip_folds
    if current_fold > get_the_lastest_fold(
            hparams) or current_fold not in folds:
        return False
    return True
