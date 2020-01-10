# import train_custom
# import test_model
import visualisations

def main():
    BATCH=100
    EPOCHS=50
    region="50S69W"
    model_name = "tadv_tphys"
    model_name = "qadv_qphys"
    model_name = "qtot_qphys"
    model_name = "qcomb_add_dot_qphys_deep"
    model_name = "qcomb_add_dot_qloss_qphys_deep"
    # model_name = "tcomb_tphys"
    # train_custom.create_train_model(BATCH, EPOCHS, model_name=model_name)

    history = 'data/models/history/model_{0}_epochs_{1}.history'.format(model_name, EPOCHS)
    visualisations.visualise_history(history)
    
    # print("Testing the model")
    # test_model.test_model(model_name=model_name)

    # Visualise
    pred_file = "{0}_{1}.hdf5".format(model_name, region)
    figname = "{0}_{1}.png".format(model_name, region)
    visualisations.visualise_predictions_q(pred_file, figname)
    
    
    # np_file = 'qphys_predict_manual.npz'
    # figname = "qadv_predict_manual.png"
    # np_file = 'qphys_predict_qadv.npz'
    # figname = "qphys_predict_qadv.png"
    # np_file = 'data/models/predictions/qtot_predict.npz'
    # figname = "data/models/predictions/qphys_predict_qtot.png"
    # visualisations.visualise_predictions_q(np_file, figname)

    np_file = 'data/models/predictions/tcomb_predict_norm.npz'
    figname = "data/models/predictions/tcomb_predict_norm.png"
    # visualisations.visualise_predictions_t(np_file, figname)

    np_file = 'data/models/predictions/qT_predict.npz'
    figname = "data/models/predictions/qT_predict.png"
    # visualisations.visualise_predictions_qT(np_file, figname)
    
if __name__ == "__main__":
    main()
