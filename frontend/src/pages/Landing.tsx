import Hero from '../components/Hero';

interface LandingProps {
  onStart: () => void;
}

const Landing = ({ onStart }: LandingProps) => {
  return (
    <div className="relative min-h-[90vh] flex flex-col justify-center">
      <Hero onStart={onStart} />
    </div>
  );
};

export default Landing;
