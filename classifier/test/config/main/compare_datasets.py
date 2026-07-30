from src.classifier.task import Main

class CompareDatasets(Main):
    argparser = Main.argparser

    def run(self):
        from bbreww.classifier.test.config.analysis.compare import VerifyOutput
        
        # Instantiate and run verification
        verify = VerifyOutput()
        verify.parse(self.args)
        verify.run()
